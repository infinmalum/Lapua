package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

const testImgData = "cG5nLWJ5dGVz" // base64 of "png-bytes"

// The Anthropic adapter embeds the neutral ContentBlock straight into
// anthropicReq, so an image/document block must reach the wire verbatim as the
// native {"type":"image","source":{...}} shape — including when nested inside a
// tool_result's Content.
func TestAnthropicImageBlockPassthrough(t *testing.T) {
	req := CompletionRequest{
		Messages: []Message{
			{Role: RoleUser, Content: []ContentBlock{TextBlock("look"), ImageBlock(MediaTypePNG, testImgData)}},
			{Role: RoleAssistant, Content: []ContentBlock{{Type: BlockToolUse, ID: "c1", Name: "Read", Input: json.RawMessage(`{"file_path":"a.png"}`)}}},
			{Role: RoleUser, Content: []ContentBlock{{
				Type: BlockToolResult, ToolUseID: "c1",
				Content: []ContentBlock{TextBlock("attached"), DocumentBlock(MediaTypePDF, testImgData)},
			}}},
		},
	}
	raw, m := captureReq(t, Config{Format: FormatAnthropic}, req, sse(`data: {"type":"message_stop"}`, ``))

	wantImg := `"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + testImgData + `"}`
	if !strings.Contains(raw, wantImg) {
		t.Fatalf("image block not on the Anthropic wire verbatim:\n%s", raw)
	}
	wantDoc := `"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + testImgData + `"}`
	if !strings.Contains(raw, wantDoc) {
		t.Fatalf("document block not on the Anthropic wire verbatim:\n%s", raw)
	}
	msgs, _ := m["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages=%d, want 3", len(msgs))
	}
}

// A user message mixing text and image blocks must serialize as a
// content-parts array, with images as data URIs and PDFs as file parts.
func TestToOpenAIMessagesUserMedia(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []ContentBlock{TextBlock("look"), ImageBlock(MediaTypePNG, testImgData)}},
		{Role: RoleUser, Content: []ContentBlock{TextBlock("report"), DocumentBlock(MediaTypePDF, testImgData)}},
	}
	out := toOpenAIMessages("", msgs)
	if len(out) != 2 {
		t.Fatalf("messages=%d, want 2", len(out))
	}

	parts, ok := out[0].Content.([]oaContentPart)
	if !ok {
		t.Fatalf("message0 content = %T, want parts array", out[0].Content)
	}
	if len(parts) != 2 || parts[0].Type != "text" || parts[0].Text != "look" {
		t.Fatalf("parts0 = %+v", parts)
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil ||
		parts[1].ImageURL.URL != "data:image/png;base64,"+testImgData {
		t.Fatalf("parts1 = %+v", parts[1])
	}

	parts, ok = out[1].Content.([]oaContentPart)
	if !ok || len(parts) != 2 || parts[0].Text != "report" || parts[1].Type != "file" ||
		parts[1].File == nil || parts[1].File.Filename != "document.pdf" ||
		parts[1].File.FileData != "data:application/pdf;base64,"+testImgData {
		t.Fatalf("message1 content = %+v", out[1].Content)
	}
}

// Media nested in a tool_result is promoted to a synthetic user message after
// the role:tool message, whose content must stay a string.
func TestToOpenAIMessagesToolResultPromotion(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []ContentBlock{{Type: BlockToolUse, ID: "c1", Name: "Read", Input: json.RawMessage(`{}`)}}},
		{Role: RoleUser, Content: []ContentBlock{{
			Type: BlockToolResult, ToolUseID: "c1",
			Content: []ContentBlock{TextBlock("res text"), ImageBlock(MediaTypeJPEG, testImgData)},
		}}},
	}
	out := toOpenAIMessages("", msgs)
	if len(out) != 3 {
		t.Fatalf("messages=%d, want [assistant, tool, promoted-user]: %+v", len(out), out)
	}
	if out[1].Role != "tool" || out[1].ToolCallID != "c1" {
		t.Fatalf("message1 = %+v", out[1])
	}
	if s, _ := out[1].Content.(string); s != "res text" {
		t.Fatalf("tool content = %v, want string", out[1].Content)
	}
	parts, ok := out[2].Content.([]oaContentPart)
	if !ok || len(parts) != 2 {
		t.Fatalf("promoted content = %+v", out[2].Content)
	}
	if parts[0].Text != toolMediaAnchor {
		t.Fatalf("anchor = %q", parts[0].Text)
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL.URL != "data:image/jpeg;base64,"+testImgData {
		t.Fatalf("promoted image part = %+v", parts[1])
	}
}

// An image-only user message must not vanish (Content is a parts array, not an
// empty string).
func TestToOpenAIMessagesImageOnlyUserMessage(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []ContentBlock{ImageBlock(MediaTypePNG, testImgData)}},
	}
	out := toOpenAIMessages("", msgs)
	if len(out) != 1 {
		t.Fatalf("messages=%d, want 1 (image-only message must survive)", len(out))
	}
	parts, ok := out[0].Content.([]oaContentPart)
	if !ok || len(parts) != 1 || parts[0].Type != "image_url" {
		t.Fatalf("content = %+v", out[0].Content)
	}
}

// Responses sends media on the function result associated with each call.
func TestToResponsesInputMedia(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []ContentBlock{TextBlock("look"), ImageBlock(MediaTypePNG, testImgData)}},
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: BlockToolUse, ID: "c1", Name: "Read", Input: json.RawMessage(`{}`)},
			{Type: BlockToolUse, ID: "c2", Name: "Read", Input: json.RawMessage(`{}`)},
		}},
		{Role: RoleUser, Content: []ContentBlock{
			{Type: BlockToolResult, ToolUseID: "c1", Content: []ContentBlock{DocumentBlock(MediaTypePDF, testImgData)}},
			{Type: BlockToolResult, ToolUseID: "c2", Content: []ContentBlock{
				TextBlock("preview"), ImageBlock(MediaTypeJPEG, testImgData), TextBlock("end"),
			}},
		}},
	}
	in := toResponsesInput(msgs)
	if len(in) != 5 {
		t.Fatalf("items=%d: %+v", len(in), in)
	}
	parts, ok := in[0].Content.([]orContentPart)
	if !ok || len(parts) != 2 {
		t.Fatalf("item0 content = %+v", in[0].Content)
	}
	if parts[0].Type != "input_text" || parts[0].Text != "look" {
		t.Fatalf("part0 = %+v", parts[0])
	}
	if parts[1].Type != "input_image" || parts[1].ImageURL != "data:image/png;base64,"+testImgData {
		t.Fatalf("part1 = %+v", parts[1])
	}
	if in[3].Type != "function_call_output" || in[3].CallID != "c1" ||
		in[4].Type != "function_call_output" || in[4].CallID != "c2" {
		t.Fatalf("function call outputs = %+v", in[3:])
	}
	pdf, ok := in[3].Output.([]orContentPart)
	if !ok || len(pdf) != 1 || pdf[0].Type != "input_file" || pdf[0].Filename != "document.pdf" ||
		pdf[0].FileData != "data:application/pdf;base64,"+testImgData {
		t.Fatalf("PDF output = %+v", in[3].Output)
	}
	image, ok := in[4].Output.([]orContentPart)
	if !ok || len(image) != 3 || image[0].Text != "preview" || image[1].Type != "input_image" ||
		image[1].ImageURL != "data:image/jpeg;base64,"+testImgData || image[2].Text != "end" {
		t.Fatalf("image output = %+v", in[4].Output)
	}
}

func TestMediaOnlyToolResultRequestBodies(t *testing.T) {
	req := CompletionRequest{Messages: []Message{
		UserText("Read both files"),
		{Role: RoleAssistant, Content: []ContentBlock{
			{Type: BlockToolUse, ID: "call_image", Name: "Read", Input: json.RawMessage(`{}`)},
			{Type: BlockToolUse, ID: "call_pdf", Name: "Read", Input: json.RawMessage(`{}`)},
		}},
		{Role: RoleUser, Content: []ContentBlock{
			{Type: BlockToolResult, ToolUseID: "call_image", Content: []ContentBlock{ImageBlock(MediaTypePNG, testImgData)}},
			{Type: BlockToolResult, ToolUseID: "call_pdf", Content: []ContentBlock{DocumentBlock(MediaTypePDF, testImgData)}},
		}},
	}}
	chat := &openaiProvider{}
	responses := &openaiResponsesProvider{}
	for _, stream := range []bool{false, true} {
		chatBody, err := chat.buildBody(req, stream)
		if err != nil {
			t.Fatal(err)
		}
		var chatReq struct {
			Messages []struct {
				Role    string
				Content json.RawMessage
				CallID  string `json:"tool_call_id"`
			}
		}
		if err := json.Unmarshal(chatBody, &chatReq); err != nil {
			t.Fatal(err)
		}
		msgs := chatReq.Messages
		if len(msgs) != 5 || msgs[2].Role != "tool" || msgs[2].CallID != "call_image" ||
			string(msgs[2].Content) != `""` || msgs[3].Role != "tool" ||
			msgs[3].CallID != "call_pdf" || string(msgs[3].Content) != `""` ||
			msgs[4].Role != "user" || !strings.Contains(string(msgs[4].Content), `"image_url":`) ||
			!strings.Contains(string(msgs[4].Content), `"file":`) {
			t.Fatalf("Chat stream=%v: malformed tool replies: %s", stream, chatBody)
		}

		responsesBody, err := responses.buildBody(req, stream)
		if err != nil {
			t.Fatal(err)
		}
		var responsesReq struct {
			Input []struct {
				Type   string
				CallID string `json:"call_id"`
				Output []orContentPart
			}
		}
		if err := json.Unmarshal(responsesBody, &responsesReq); err != nil {
			t.Fatal(err)
		}
		items := responsesReq.Input
		if len(items) != 5 || items[3].Type != "function_call_output" || items[3].CallID != "call_image" ||
			len(items[3].Output) != 1 || items[3].Output[0].Type != "input_image" ||
			items[4].Type != "function_call_output" || items[4].CallID != "call_pdf" ||
			len(items[4].Output) != 1 || items[4].Output[0].Type != "input_file" {
			t.Fatalf("Responses stream=%v: malformed tool replies: %s", stream, responsesBody)
		}
	}
}

func TestAttachmentSources(t *testing.T) {
	const imageURL = "https://example.com/diagram.png"
	const pdfURL = "https://example.com/report.pdf"
	image := ContentBlock{Type: BlockImage, Source: &BlockSource{
		Type: "url", URL: imageURL, MediaType: MediaTypePNG, Detail: "high",
	}}
	imageID := ContentBlock{Type: BlockImage, Source: &BlockSource{Type: "file", FileID: "file_image"}}
	pdf := ContentBlock{Type: BlockDocument, Source: &BlockSource{
		Type: "url", URL: pdfURL, MediaType: MediaTypePDF, Detail: "low",
	}}
	pdfID := ContentBlock{Type: BlockDocument, Source: &BlockSource{
		Type: "file", FileID: "file_report", MediaType: MediaTypePDF,
	}}

	chat := toOpenAIMessages("", []Message{{Role: RoleUser, Content: []ContentBlock{image, pdfID, pdf, imageID}}})
	chatParts, ok := chat[0].Content.([]oaContentPart)
	if !ok || len(chatParts) != 4 ||
		chatParts[0].ImageURL == nil || *chatParts[0].ImageURL != (oaImageURL{URL: imageURL, Detail: "high"}) ||
		chatParts[1].File == nil || *chatParts[1].File != (oaFile{FileID: "file_report"}) ||
		chatParts[2].Type != "text" || chatParts[3].Type != "text" {
		t.Fatalf("Chat parts = %+v", chat[0].Content)
	}

	responses := toResponsesInput([]Message{{Role: RoleUser, Content: []ContentBlock{image, imageID, pdf, pdfID}}})
	responseParts, ok := responses[0].Content.([]orContentPart)
	if !ok || len(responseParts) != 4 ||
		responseParts[0] != (orContentPart{Type: "input_image", ImageURL: imageURL, Detail: "high"}) ||
		responseParts[1] != (orContentPart{Type: "input_image", FileID: "file_image"}) ||
		responseParts[2] != (orContentPart{Type: "input_file", FileURL: pdfURL, Detail: "low"}) ||
		responseParts[3] != (orContentPart{Type: "input_file", FileID: "file_report"}) {
		t.Fatalf("Responses parts = %+v", responses[0].Content)
	}

	req := CompletionRequest{Messages: []Message{
		{Role: RoleUser, Content: []ContentBlock{image, pdf}},
		{Role: RoleAssistant, Content: []ContentBlock{{Type: BlockToolUse, ID: "c1", Name: "Read"}}},
		{Role: RoleUser, Content: []ContentBlock{{
			Type: BlockToolResult, ToolUseID: "c1",
			Content: []ContentBlock{imageID, pdfID, DocumentBlock(MediaTypeDOCX, testImgData)},
		}}},
	}}
	p := &anthropicProvider{}
	body, err := p.buildBody(req, false)
	if err != nil {
		t.Fatal(err)
	}
	var claude anthropicReq
	if err := json.Unmarshal(body, &claude); err != nil {
		t.Fatal(err)
	}
	user := claude.Messages[0].Content
	tool := claude.Messages[2].Content[0].Content
	if *user[0].Source != (BlockSource{Type: "url", URL: imageURL}) ||
		*user[1].Source != (BlockSource{Type: "url", URL: pdfURL}) ||
		*tool[0].Source != (BlockSource{Type: "file", FileID: "file_image"}) ||
		*tool[1].Source != (BlockSource{Type: "file", FileID: "file_report"}) ||
		tool[2].Type != BlockText || tool[2].Source != nil {
		t.Fatalf("Claude attachments = %s", body)
	}
	if image.Source.Detail != "high" || pdfID.Source.MediaType != MediaTypePDF ||
		req.Messages[2].Content[0].Content[2].Type != BlockDocument {
		t.Fatal("Claude request changed the original history")
	}
}

func TestResponsesOfficeAttachments(t *testing.T) {
	for _, tc := range []struct{ mime, filename string }{
		{MediaTypeDOCX, "brief.docx"},
		{MediaTypeXLSX, "budget.xlsx"},
		{MediaTypePPTX, "slides.pptx"},
	} {
		t.Run(tc.filename, func(t *testing.T) {
			doc := DocumentBlock(tc.mime, testImgData)
			doc.Source.Filename = tc.filename
			msg := Message{Role: RoleUser, Content: []ContentBlock{doc}}
			responses := toResponsesInput([]Message{msg})
			parts, ok := responses[0].Content.([]orContentPart)
			if !ok || len(parts) != 1 || parts[0] != (orContentPart{
				Type: "input_file", Filename: tc.filename, FileData: "data:" + tc.mime + ";base64," + testImgData,
			}) {
				t.Fatalf("Responses parts = %+v", responses[0].Content)
			}
			chat := toOpenAIMessages("", []Message{msg})
			if notice, ok := chat[0].Content.(string); !ok || !strings.Contains(notice, "omitted") ||
				strings.Contains(notice, testImgData) {
				t.Fatalf("Chat content = %+v", chat[0].Content)
			}
		})
	}
}

// Source must round-trip through JSON verbatim — transcripts persist
// llm.Message with encoding/json, so a recorded session with images must
// resume with identical payloads.
func TestContentBlockMediaJSONRoundTrip(t *testing.T) {
	orig := ImageBlock(MediaTypeWEBP, testImgData)
	raw, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back ContentBlock
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Type != BlockImage || back.Source == nil ||
		back.Source.MediaType != MediaTypeWEBP || back.Source.Data != testImgData ||
		back.Source.Type != "base64" {
		t.Fatalf("round trip lost media: %+v (%s)", back, raw)
	}
	if !strings.Contains(string(raw), `"source"`) {
		t.Fatalf("source key missing: %s", raw)
	}
}
