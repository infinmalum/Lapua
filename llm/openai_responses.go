package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
)

// openaiResponsesProvider speaks the OpenAI Responses API (POST /v1/responses),
// a different wire protocol from Chat Completions: the system prompt is a
// top-level `instructions`, history is a flat `input` array of items
// (message / function_call / function_call_output), tools are flattened, the
// output cap is `max_output_tokens`, reasoning strength is `reasoning.effort`,
// and streaming uses many named SSE events (response.output_text.delta, …)
// instead of choices[].delta. It translates all of that to and from the neutral
// content-block model so the harness is unchanged.
//
// It is stateless: store is false and the full history is sent every call, so it
// composes with the host's own transcript/compaction (no previous_response_id).
type openaiResponsesProvider struct{ cfg Config }

// ---- request ----

type orReq struct {
	Model           string        `json:"model"`
	Instructions    string        `json:"instructions,omitempty"`
	Input           []orInputItem `json:"input"`
	Tools           []orTool      `json:"tools,omitempty"`
	MaxOutputTokens int           `json:"max_output_tokens,omitempty"`
	Reasoning       *orReasoning  `json:"reasoning,omitempty"`
	Temperature     *float64      `json:"temperature,omitempty"`
	Stream          bool          `json:"stream"`
	Store           bool          `json:"store"`
}

type orReasoning struct {
	Effort string `json:"effort"`
}

// orInputItem is a union: a message (Role+Content, no Type) or a typed
// function_call / function_call_output item. Content is a string or, for a
// message carrying media, a parts array. Output uses the same string-or-parts
// union for function results. A nil interface omits the field, while an
// interface holding an empty string still serializes the required output.
type orInputItem struct {
	Type      string `json:"type,omitempty"`
	Role      string `json:"role,omitempty"`
	Content   any    `json:"content,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    any    `json:"output,omitempty"`
}

// orContentPart is one entry of a message item's content-parts array — the
// Responses API multimodal input shape.
type orContentPart struct {
	Type     string `json:"type"` // "input_text" | "input_image" | "input_file"
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
	Filename string `json:"filename,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

type orTool struct {
	Type        string         `json:"type"` // "function"
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

func (p *openaiResponsesProvider) buildBody(req CompletionRequest, stream bool) ([]byte, error) {
	body := orReq{
		Model:           p.cfg.Model,
		Instructions:    joinSystem(req.System),
		Input:           toResponsesInput(req.Messages),
		MaxOutputTokens: req.MaxTokens,
		Temperature:     req.Temperature,
		Stream:          stream,
		Store:           false, // stateless: full history sent each call
	}
	// reasoning.effort is the Responses equivalent of reasoning_effort. A
	// per-request "disabled" (compaction's summary call) suppresses it; otherwise
	// send it only when configured.
	if req.Thinking != "disabled" && p.cfg.ReasoningEffort != "" {
		body.Reasoning = &orReasoning{Effort: p.cfg.ReasoningEffort}
	}
	for _, t := range req.Tools {
		body.Tools = append(body.Tools, orTool{Type: "function", Name: t.Name, Description: t.Description, Parameters: t.InputSchema})
	}
	return json.Marshal(body)
}

// toResponsesInput flattens neutral messages into Responses input items: text
// becomes a role message (string content, or a content-parts array when the
// message carries image/document blocks), tool_use becomes a function_call, and
// tool_result becomes a function_call_output whose output carries text or
// media parts bound to the call ID. Thinking blocks are
// dropped on replay (mirrors the Chat Completions adapter's default and avoids
// echoing provider-specific reasoning state the Responses API manages itself).
func toResponsesInput(msgs []Message) []orInputItem {
	var out []orInputItem
	for _, m := range msgs {
		role := string(m.Role)
		var text strings.Builder
		var parts []orContentPart
		hasMedia := false
		// flush emits accumulated content as a role message, preserving block
		// order: text/media preceding a tool_use/tool_result must appear before
		// that item in the input array, not after it. A message with media is
		// emitted as a content-parts array; text-only stays a plain string.
		flush := func() {
			if hasMedia {
				if s := text.String(); s != "" {
					parts = append(parts, orContentPart{Type: "input_text", Text: s})
					text.Reset()
				}
				if len(parts) > 0 {
					out = append(out, orInputItem{Role: role, Content: parts})
					parts = nil
				}
				return
			}
			if s := text.String(); s != "" {
				out = append(out, orInputItem{Role: role, Content: s})
				text.Reset()
			}
		}
		addText := func(s string) {
			if s == "" {
				return
			}
			if hasMedia {
				parts = append(parts, orContentPart{Type: "input_text", Text: s})
				return
			}
			text.WriteString(s)
		}
		enterMedia := func() {
			if !hasMedia {
				hasMedia = true
				if s := text.String(); s != "" {
					parts = append(parts, orContentPart{Type: "input_text", Text: s})
					text.Reset()
				}
			}
		}
		for _, b := range m.Content {
			switch b.Type {
			case BlockText:
				addText(b.Text)
			case BlockImage, BlockDocument:
				part := orMediaPart(b)
				if part.Type == "input_text" {
					addText(part.Text)
					continue
				}
				enterMedia()
				parts = append(parts, part)
			case BlockToolUse:
				flush()
				args := string(b.Input)
				if strings.TrimSpace(args) == "" {
					args = "{}"
				}
				out = append(out, orInputItem{Type: "function_call", CallID: b.ID, Name: b.Name, Arguments: args})
			case BlockToolResult:
				flush()
				out = append(out, orInputItem{Type: "function_call_output", CallID: b.ToolUseID, Output: orToolOutput(b.Content)})
			}
		}
		flush()
	}
	return out
}

func orToolOutput(blocks []ContentBlock) any {
	var parts []orContentPart
	hasMedia := false
	for _, b := range blocks {
		switch b.Type {
		case BlockText:
			if b.Text != "" {
				parts = append(parts, orContentPart{Type: "input_text", Text: b.Text})
			}
		case BlockImage, BlockDocument:
			part := orMediaPart(b)
			hasMedia = hasMedia || part.Type != "input_text"
			parts = append(parts, part)
		}
	}
	if hasMedia {
		return parts
	}
	var text strings.Builder
	for _, part := range parts {
		text.WriteString(part.Text)
	}
	return text.String()
}

func orMediaPart(b ContentBlock) orContentPart {
	s := b.Source
	if s != nil {
		switch b.Type {
		case BlockImage:
			switch {
			case s.Type == "base64" && s.MediaType != "" && s.Data != "":
				return orContentPart{Type: "input_image", ImageURL: s.dataURI(), Detail: s.Detail}
			case s.Type == "url" && s.URL != "":
				return orContentPart{Type: "input_image", ImageURL: s.URL, Detail: s.Detail}
			case s.Type == "file" && s.FileID != "":
				return orContentPart{Type: "input_image", FileID: s.FileID, Detail: s.Detail}
			}
		case BlockDocument:
			var detail string
			if s.MediaType == "" || s.MediaType == MediaTypePDF {
				detail = s.Detail
			}
			switch {
			case s.Type == "base64" && s.MediaType != "" && s.Data != "" && s.documentFilename() != "":
				return orContentPart{Type: "input_file", FileData: s.dataURI(), Filename: s.documentFilename(), Detail: detail}
			case s.Type == "url" && s.URL != "":
				return orContentPart{Type: "input_file", FileURL: s.URL, Detail: detail}
			case s.Type == "file" && s.FileID != "":
				return orContentPart{Type: "input_file", FileID: s.FileID, Detail: detail}
			}
		}
	}
	return orContentPart{Type: "input_text", Text: mediaNotice(b)}
}

func (p *openaiResponsesProvider) setHeaders(r *http.Request) {
	r.Header.Set("content-type", "application/json")
	r.Header.Set("authorization", "Bearer "+p.cfg.APIKey)
}

// ---- streaming ----

func (p *openaiResponsesProvider) Stream(ctx context.Context, req CompletionRequest) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		body, err := p.buildBody(req, true)
		if err != nil {
			yield(StreamEvent{}, err)
			return
		}
		resp, err := doStream(ctx, p.cfg, p.cfg.BaseURL+"/responses", body, func(r *http.Request) {
			p.setHeaders(r)
			r.Header.Set("accept", "text/event-stream")
		}, "openai-responses")
		if err != nil {
			yield(StreamEvent{}, err)
			return
		}
		defer resp.Body.Close()
		scan := newSSEScanner(resp.Body)
		for {
			event, data, serr := scan.next()
			if serr == io.EOF {
				yield(StreamEvent{}, fmt.Errorf("openai-responses: %w", io.ErrUnexpectedEOF))
				return
			}
			if serr != nil {
				yield(StreamEvent{}, serr)
				return
			}
			if data == "" {
				continue
			}
			ev, ok, perr := parseResponsesFrame(event, data)
			if perr != nil {
				yield(StreamEvent{}, perr)
				return
			}
			if !ok {
				continue
			}
			if !yield(ev, nil) || ev.Type == SEMessageDelta && ev.StopReason != "" {
				return
			}
		}
	}
}

// parseResponsesFrame maps one named Responses SSE event to a neutral
// StreamEvent. Unrecognized events (response.created, *.in_progress, item.done,
// content_part.*, etc.) return ok=false and are skipped.
func parseResponsesFrame(event, data string) (StreamEvent, bool, error) {
	// The event name is authoritative; some gateways omit the SSE `event:` line
	// and put the discriminator only in the JSON `type` field, so fall back to it.
	var f struct {
		Type        string `json:"type"`
		OutputIndex *int   `json:"output_index"`
		// output_item.added: item carries the function_call id/name.
		Item struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"item"`
		Delta    string `json:"delta"` // *_text.delta / function_call_arguments.delta
		Response struct {
			Status            string `json:"status"`
			IncompleteDetails *struct {
				Reason string `json:"reason"`
			} `json:"incomplete_details"`
			Usage *orUsage `json:"usage"`
		} `json:"response"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(data), &f); err != nil {
		return StreamEvent{}, false, nil // tolerate keepalives / non-JSON
	}
	kind := event
	if kind == "" {
		kind = f.Type
	}
	index := 0
	if f.OutputIndex != nil {
		index = *f.OutputIndex
	}
	switch kind {
	case "response.output_item.added":
		if f.Item.Type == "function_call" {
			return StreamEvent{Type: SEToolUseStart, ToolID: f.Item.CallID, ToolName: f.Item.Name, ToolIndex: index}, true, nil
		}
	case "response.output_text.delta":
		if f.Delta != "" {
			return StreamEvent{Type: SETextDelta, Text: f.Delta}, true, nil
		}
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		if f.Delta != "" {
			return StreamEvent{Type: SEThinkingDelta, Text: f.Delta}, true, nil
		}
	case "response.function_call_arguments.delta":
		if f.Delta != "" {
			return StreamEvent{Type: SEToolInputJSON, Text: f.Delta, ToolIndex: index, HasToolIndex: f.OutputIndex != nil}, true, nil
		}
	case "response.completed", "response.incomplete":
		ev := StreamEvent{Type: SEMessageDelta, StopReason: "end_turn"}
		if f.Response.IncompleteDetails != nil && f.Response.IncompleteDetails.Reason == "max_output_tokens" {
			ev.StopReason = "max_tokens"
		}
		if f.Response.Usage != nil {
			ev.Usage = f.Response.Usage.norm()
		}
		return ev, true, nil
	case "response.failed", "error":
		msg := "stream error"
		if f.Error != nil {
			msg = f.Error.Message
		}
		return StreamEvent{}, false, fmt.Errorf("openai-responses: %s", msg)
	}
	return StreamEvent{}, false, nil
}

// ---- non-streaming ----

func (p *openaiResponsesProvider) Complete(ctx context.Context, req CompletionRequest) (Message, string, Usage, error) {
	body, err := p.buildBody(req, false)
	if err != nil {
		return Message{}, "", Usage{}, err
	}
	resp, err := doStream(ctx, p.cfg, p.cfg.BaseURL+"/responses", body, func(r *http.Request) {
		p.setHeaders(r)
		r.Header.Set("accept", "application/json")
	}, "openai-responses")
	if err != nil {
		return Message{}, "", Usage{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return Message{}, "", Usage{}, err
	}
	return parseResponsesResponse(raw)
}

type orUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
}

// norm maps Responses usage to the canonical Usage. InputTokens already counts
// the full input including cached tokens (OpenAI semantics); CacheReadTokens is
// the cached subset.
func (u orUsage) norm() Usage {
	return Usage{
		InputTokens:     u.InputTokens,
		OutputTokens:    u.OutputTokens,
		CacheReadTokens: u.InputTokensDetails.CachedTokens,
	}
}

// parseResponsesResponse decodes a non-streaming /v1/responses body into the
// neutral Message: output message text → text block, reasoning summary →
// thinking block, function_call → tool_use block (ID = call_id).
func parseResponsesResponse(raw []byte) (Message, string, Usage, error) {
	var r struct {
		Output []struct {
			Type    string `json:"type"`
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			Summary []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"summary"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"output"`
		Status            string `json:"status"`
		IncompleteDetails *struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
		Usage *orUsage `json:"usage"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return Message{}, "", Usage{}, fmt.Errorf("openai-responses: decode response: %w (body: %s)", err, truncate(string(raw), 500))
	}
	if r.Error != nil {
		return Message{}, "", Usage{}, fmt.Errorf("openai-responses: %s", r.Error.Message)
	}
	var usage Usage
	if r.Usage != nil {
		usage = r.Usage.norm()
	}
	var blocks []ContentBlock
	hasToolCall := false
	for _, item := range r.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" && c.Text != "" {
					blocks = append(blocks, TextBlock(c.Text))
				}
			}
		case "reasoning":
			// A reasoning item carries its thinking in `summary` (summary_text parts)
			// and/or in `content` (reasoning_text parts). The streaming path already
			// accepts both — response.reasoning_summary_text.delta and
			// response.reasoning_text.delta — so read both here too; taking only
			// `summary` drops the whole turn's thinking on models that fill `content`.
			var think strings.Builder
			for _, s := range item.Summary {
				think.WriteString(s.Text)
			}
			for _, c := range item.Content {
				if c.Type == "reasoning_text" {
					think.WriteString(c.Text)
				}
			}
			if think.Len() > 0 {
				blocks = append(blocks, ContentBlock{Type: BlockThinking, Thinking: think.String()})
			}
		case "function_call":
			args := item.Arguments
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			blocks = append(blocks, ContentBlock{Type: BlockToolUse, ID: item.CallID, Name: item.Name, Input: []byte(args)})
			hasToolCall = true
		}
	}
	stop := "end_turn"
	if hasToolCall {
		stop = "tool_use"
	}
	if r.IncompleteDetails != nil && r.IncompleteDetails.Reason == "max_output_tokens" {
		stop = "max_tokens"
	}
	return Message{Role: RoleAssistant, Content: blocks}, stop, usage, nil
}
