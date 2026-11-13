package permission

import (
	"context"
	"encoding/json"
	"testing"
)

func eval(req Request) Behavior {
	return Evaluate(context.Background(), req).Behavior
}

func TestDenyPriorityOverBypass(t *testing.T) {
	req := Request{ToolName: "Bash", Ctx: Context{Mode: ModeBypass, Disallowed: []string{"Bash"}}}
	if b := eval(req); b != Deny {
		t.Fatalf("deny must beat bypass, got %v", b)
	}
}

func TestReadOnlyAutoAllow(t *testing.T) {
	if b := eval(Request{ToolName: "Read", IsReadOnly: true, Ctx: Context{Mode: ModeDefault}}); b != Allow {
		t.Fatalf("read-only should allow, got %v", b)
	}
}

func TestPlanModeDeniesWrites(t *testing.T) {
	if b := eval(Request{ToolName: "Write", IsReadOnly: false, Ctx: Context{Mode: ModePlan}}); b != Deny {
		t.Fatalf("plan mode should deny writes, got %v", b)
	}
}

func TestAcceptEditsAllowsEdit(t *testing.T) {
	if b := eval(Request{ToolName: "Edit", Ctx: Context{Mode: ModeAcceptEdits}}); b != Allow {
		t.Fatalf("acceptEdits should allow Edit, got %v", b)
	}
}

func TestAskReachesCallback(t *testing.T) {
	called := false
	req := Request{
		ToolName: "Bash", IsReadOnly: false,
		ToolDecision: AskUser("?"),
		Ctx:          Context{Mode: ModeDefault},
		Ask: func(_ context.Context, _ string, _ json.RawMessage, _ Context) Decision {
			called = true
			return Allowed()
		},
	}
	if b := eval(req); b != Allow || !called {
		t.Fatalf("ask callback not honored: b=%v called=%v", b, called)
	}
}

func TestUnsupportedSelectorFailsClosed(t *testing.T) {
	req := Request{ToolName: "Bash", Input: json.RawMessage(`{"command":"echo test"}`),
		Ctx: Context{Mode: ModeBypass, Allowed: []string{"Bash(git:*)"}}}
	if b := eval(req); b != Deny {
		t.Fatalf("unsupported selector must not grant access: %v", b)
	}
	req.Ctx.Allowed = []string{"Bash"}
	if b := eval(req); b != Allow {
		t.Fatalf("bare tool name must still match: %v", b)
	}
}

func TestNestedPolicyCannotLoosenParent(t *testing.T) {
	req := Request{ToolName: "Write", Ctx: Context{Mode: ModeBypass}}
	ctx := WithParentPolicy(t.Context(), Context{Mode: ModeDontAsk}, nil)
	if d := Evaluate(ctx, req); d.Behavior != Deny {
		t.Fatalf("child bypass loosened parent policy: %+v", d)
	}
	ctx = WithParentPolicy(t.Context(), Context{Mode: ModeBypass, Disallowed: []string{"Write"}}, nil)
	if d := Evaluate(ctx, req); d.Behavior != Deny {
		t.Fatalf("child bypass overrode explicit parent denial: %+v", d)
	}
}
