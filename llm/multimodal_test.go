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
// content-parts array (text part first, image part as a data URI); a
// document block degrades to a text notice (Chat Completions has no native
// document input).
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

	s, ok := out[1].Content.(string)
	if !ok {
		t.Fatalf("message1 content = %T, want string", out[1].Content)
	}
	if !strings.Contains(s, "[document application/pdf omitted") {
		t.Fatalf("document notice missing: %q", s)
	}
	if strings.Contains(s, testImgData) {
		t.Fatal("document payload must not reach the Chat Completions wire")
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

// Responses API: images → input_image parts, documents → input_file parts;
// tool-result media is promoted into a following user message item.
func TestToResponsesInputMedia(t *testing.T) {
	msgs := []Message{
		{Role: RoleUser, Content: []ContentBlock{TextBlock("look"), ImageBlock(MediaTypePNG, testImgData)}},
		{Role: RoleAssistant, Content: []ContentBlock{{Type: BlockToolUse, ID: "c1", Name: "Read", Input: json.RawMessage(`{}`)}}},
		{Role: RoleUser, Content: []ContentBlock{{
			Type: BlockToolResult, ToolUseID: "c1",
			Content: []ContentBlock{DocumentBlock(MediaTypePDF, testImgData)},
		}}},
	}
	in := toResponsesInput(msgs)
	// user-parts, function_call, function_call_output, promoted user.
	if len(in) != 4 {
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
	if in[3].Type != "function_call_output" && in[3].Role != "user" {
		t.Fatalf("item3 should be the promoted user message: %+v", in[3])
	}
	promoted, ok := in[3].Content.([]orContentPart)
	if !ok || len(promoted) != 2 || promoted[1].Type != "input_file" ||
		promoted[1].FileData != "data:application/pdf;base64,"+testImgData {
		t.Fatalf("promoted parts = %+v", in[3].Content)
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
