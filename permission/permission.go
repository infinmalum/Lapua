// Package permission is the agent's safety boundary. It models tool-call
// decisions (allow / deny / ask), six permission modes, allow/deny rule lists
// with deny-priority, and a host-provided interactive callback. The four-stage
// pipeline in Evaluate combines a tool's own CheckPermissions verdict with
// policy and mode to produce a final decision.
package permission

import (
	"context"
	"encoding/json"
	"strings"
)

// Behavior is the kind of a permission decision.
type Behavior string

const (
	Allow Behavior = "allow"
	Deny  Behavior = "deny"
	Ask   Behavior = "ask"
)

// Update is a persistence suggestion ("always allow this") surfaced to the host
// (FR-05.6).
type Update struct {
	Type string `json:"type"` // e.g. "addAllowRule"
	Rule string `json:"rule"` // tool name or "*"
}

// Decision is the outcome of a permission check.
type Decision struct {
	Behavior     Behavior
	UpdatedInput json.RawMessage // optional rewritten input (e.g. stripped flag)
	Message      string          // reason, shown to model on deny / to user on ask
	Suggestions  []Update        // optional "always allow" suggestions
}

// Allowed / Denied / AskUser build common decisions.
func Allowed() Decision          { return Decision{Behavior: Allow} }
func Denied(msg string) Decision { return Decision{Behavior: Deny, Message: msg} }
func AskUser(msg string) Decision {
	return Decision{Behavior: Ask, Message: msg}
}

// Mode is a permission mode (FR-05.4).
type Mode string

const (
	ModeDefault     Mode = "default"           // ask for non-read-only unless a rule allows
	ModeAcceptEdits Mode = "acceptEdits"       // auto-allow file edits/writes
	ModeBypass      Mode = "bypassPermissions" // allow everything (except deny rules)
	ModePlan        Mode = "plan"              // read-only only
	ModeDontAsk     Mode = "dontAsk"           // never prompt; rules + read-only only
	ModeAuto        Mode = "auto"              // like dontAsk but allows tool's own allow verdict
)

// Context is passed to a tool's CheckPermissions and the interactive callback.
type Context struct {
	Mode       Mode
	WorkingDir string
	Allowed    []string
	Disallowed []string
}

// CanUseTool is the host's interactive approval callback (CLI prompt, GUI, or
// automated policy). It is consulted only when the pipeline reaches the "ask"
// stage (FR-05.1, C-02).
type CanUseTool func(ctx context.Context, toolName string, input json.RawMessage, pc Context) Decision

// editTools are auto-allowed under ModeAcceptEdits.
var editTools = map[string]bool{"Write": true, "Edit": true, "MultiEdit": true, "NotebookEdit": true}

// Request bundles everything Evaluate needs without importing the tool package.
type Request struct {
	ToolName     string
	Input        json.RawMessage
	IsReadOnly   bool
	ToolDecision Decision // verdict from the tool's own CheckPermissions
	Ctx          Context
	Ask          CanUseTool
}

// Evaluate runs the four-stage pipeline (FR-05.3) and returns the final
// decision. Order encodes deny-priority (FR-05.5): an explicit deny rule wins
// over every mode, including bypass.
func Evaluate(ctx context.Context, req Request) Decision {
	result := decide(req)
	if result.Behavior == Deny {
		return result
	}
	promptContext := req.Ctx
	ask := req.Ask
	for p := parentPolicy(ctx); p != nil; p = p.parent {
		parentReq := req
		parentReq.Ctx = p.context
		parentReq.Ctx.WorkingDir = req.Ctx.WorkingDir
		decision := decide(parentReq)
		if decision.Behavior == Deny {
			return decision
		}
		if decision.Behavior == Ask {
			result = decision
		}
		promptContext = parentReq.Ctx
		ask = p.ask
	}
	if result.Behavior != Ask {
		return result
	}
	if ask != nil {
		d := ask(ctx, req.ToolName, req.Input, promptContext)
		if d.Behavior == Allow || d.Behavior == Deny {
			return d
		}
		return Denied("denied by approval callback")
	}
	return Denied("denied: no approval handler configured for '" + req.ToolName + "'")
}

type policyKey struct{}

type policy struct {
	context Context
	ask     CanUseTool
	parent  *policy
}

func parentPolicy(ctx context.Context) *policy {
	p, _ := ctx.Value(policyKey{}).(*policy)
	return p
}

// WithParentPolicy carries the caller's restrictions into nested queries.
// A nested call must pass every enclosing policy. The outermost host owns
// interactive approval.
func WithParentPolicy(ctx context.Context, pc Context, ask CanUseTool) context.Context {
	return context.WithValue(ctx, policyKey{}, &policy{context: pc, ask: ask, parent: parentPolicy(ctx)})
}

// ParentContext returns the nearest enclosing query's policy, if present.
func ParentContext(ctx context.Context) (Context, bool) {
	if p := parentPolicy(ctx); p != nil {
		return p.context, true
	}
	return Context{}, false
}

func decide(req Request) Decision {
	// Stage 1 — deny rules (absolute priority).
	if invalidRule(req.Ctx.Allowed, req.ToolName) || invalidRule(req.Ctx.Disallowed, req.ToolName) {
		return Denied("denied: unsupported permission rule for '" + req.ToolName + "'; use a tool name or '*'")
	}
	if matchRule(req.Ctx.Disallowed, req.ToolName) {
		return Denied("denied: tool '" + req.ToolName + "' is disallowed by policy")
	}
	// A tool that hard-denies (e.g. failed self-validation) is also absolute.
	if req.ToolDecision.Behavior == Deny {
		return req.ToolDecision
	}

	// Stage 2 — mode short-circuits.
	switch req.Ctx.Mode {
	case ModeBypass:
		return Allowed()
	case ModePlan:
		if !req.IsReadOnly {
			return Denied("denied: plan mode permits read-only tools only")
		}
		return Allowed()
	}

	// Stage 3 — allow rules and mode-based auto-allows.
	if matchRule(req.Ctx.Allowed, req.ToolName) {
		return Allowed()
	}
	if req.Ctx.Mode == ModeAcceptEdits && editTools[req.ToolName] {
		return Allowed()
	}
	if req.IsReadOnly {
		return Allowed()
	}
	if req.ToolDecision.Behavior == Allow {
		return req.ToolDecision
	}

	// Stage 4 — interactive confirmation (only if mode permits prompting).
	switch req.Ctx.Mode {
	case ModeDontAsk:
		return Denied("denied: '" + req.ToolName + "' requires approval but mode is dontAsk")
	case ModeAuto:
		return Denied("denied: '" + req.ToolName + "' not permitted by rules in auto mode")
	}
	return AskUser("approve " + req.ToolName + "?")
}

// Argument selectors are not supported. Fail closed for the affected tool,
// including under bypass mode, so a legacy selector cannot broaden access.
func invalidRule(rules []string, toolName string) bool {
	for _, rule := range rules {
		name, _, hasSelector := strings.Cut(strings.TrimSpace(rule), "(")
		if hasSelector && (strings.TrimSpace(name) == toolName || strings.TrimSpace(name) == "*") {
			return true
		}
	}
	return false
}

func matchRule(rules []string, toolName string) bool {
	for _, r := range rules {
		r = strings.TrimSpace(r)
		if r == "*" || r == toolName {
			return true
		}
	}
	return false
}
