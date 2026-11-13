package tool

import (
	"os"
	"strings"
	"testing"
)

func TestCapResultSpillIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	tc := &ToolContext{OutputDir: dir, MaxOutputChars: 100}
	long := strings.Repeat("x", 5000)
	first := CapResult(tc, Text(long))
	if !strings.Contains(first.Flatten(), "<persisted-output>") {
		t.Fatalf("expected spill notice: %q", first.Flatten())
	}
	second := CapResult(tc, first)
	if second.Flatten() != first.Flatten() {
		t.Fatal("second cap changed the first result")
	}
	if files, _ := os.ReadDir(dir); len(files) != 1 {
		t.Fatalf("expected exactly one spill file, got %d", len(files))
	}
}

func TestCaptureOncePassesShortOutput(t *testing.T) {
	tc := &ToolContext{MaxOutputChars: 100}
	if got := CaptureOnce(tc, "ok"); got != "ok" {
		t.Fatalf("short output changed: %q", got)
	}
}
