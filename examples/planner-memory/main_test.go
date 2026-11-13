package main

import (
	"bufio"
	"strings"
	"testing"
)

func TestApprovalSharesPromptInput(t *testing.T) {
	in := bufio.NewScanner(strings.NewReader("first task\nyes\nsecond task\nn\nadd tests\n"))
	if !in.Scan() || in.Text() != "first task" {
		t.Fatal("first prompt missing")
	}
	if approved, _ := approvePlan(in, "first plan"); !approved {
		t.Fatal("first plan was not approved")
	}
	if !in.Scan() || in.Text() != "second task" {
		t.Fatal("approval consumed the next prompt")
	}
	if approved, feedback := approvePlan(in, "second plan"); approved || feedback != "add tests" {
		t.Fatalf("approval=%v feedback=%q", approved, feedback)
	}
}
