// Package memory gives the agent persistent, file-backed long-term memory.
// Each memory is a markdown file with YAML frontmatter
// (name / description / type) in a memory directory; a MEMORY.md index lists
// them. The agent recalls memories relevant to a query and records new ones via
// tools, and relevant memories can be auto-injected into the conversation.
package memory

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Type classifies a memory.
type Type string

const (
	TypeUser      Type = "user"      // facts about the user (role, preferences)
	TypeFeedback  Type = "feedback"  // guidance on how the agent should work
	TypeProject   Type = "project"   // ongoing work, goals, constraints
	TypeReference Type = "reference" // pointers to external resources
)

// Memory is a single stored fact.
type Memory struct {
	Name        string
	Description string
	Type        Type
	Content     string
	Path        string
	ModTime     time.Time
}

// Store is a memory directory.
type Store struct {
	Dir string
}

// NewStore roots a store at dir (created on first write).
func NewStore(dir string) *Store { return &Store{Dir: dir} }

const indexFile = "MEMORY.md"

func (s *Store) ensure() error { return os.MkdirAll(s.Dir, 0o755) }

var slugRe = regexp.MustCompile(`[^a-z0-9]+`)

func slug(name string) string {
	s := slugRe.ReplaceAllString(strings.ToLower(name), "-")
	return strings.Trim(s, "-")
}

// Scan reads every memory file (excluding the index), newest first.
func (s *Store) Scan() ([]Memory, error) {
	entries, err := os.ReadDir(s.Dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []Memory
	for _, e := range entries {
		if e.IsDir() || e.Name() == indexFile || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		m, err := s.readFile(filepath.Join(s.Dir, e.Name()))
		if err == nil {
			out = append(out, m)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ModTime.After(out[j].ModTime) })
	return out, nil
}

// Read resolves an exact name first and accepts the legacy slug as an alias.
func (s *Store) Read(name string) (Memory, error) {
	name = strings.TrimSpace(name)
	unique := filepath.Join(s.Dir, uniqueFilename(name))
	m, err := s.readFile(unique)
	if err == nil || !errors.Is(err, os.ErrNotExist) {
		return m, err
	}
	legacy := filepath.Join(s.Dir, slug(name)+".md")
	m, err = s.readFile(legacy)
	if err == nil && m.Name != name && name != slug(m.Name) {
		return Memory{}, fmt.Errorf("memory %q: %w", name, os.ErrNotExist)
	}
	return m, err
}

func uniqueFilename(name string) string {
	return fmt.Sprintf("memory_%x.md", sha256.Sum256([]byte(name)))
}

func (s *Store) readFile(path string) (Memory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Memory{}, err
	}
	m := parseFrontmatter(string(data))
	m.Path = path
	if m.Name == "" {
		m.Name = strings.TrimSuffix(filepath.Base(path), ".md")
	}
	if info, e := os.Stat(path); e == nil {
		m.ModTime = info.ModTime()
	}
	return m, nil
}

// Save updates a memory by exact name and rebuilds the MEMORY.md index.
// An existing legacy filename is reused only when it belongs to that name.
func (s *Store) Save(m Memory) error {
	m.Name = strings.TrimSpace(m.Name)
	if m.Name == "" || strings.ContainsAny(m.Name, "\r\n") {
		return fmt.Errorf("memory: a single-line name is required")
	}
	if err := s.ensure(); err != nil {
		return err
	}
	unique := uniqueFilename(m.Name)
	legacy := slug(m.Name)
	filename := unique
	if legacy != "" && len(legacy) <= 180 {
		if _, err := os.Stat(filepath.Join(s.Dir, unique)); errors.Is(err, os.ErrNotExist) {
			filename = legacy + ".md"
		}
	}
	// Also reuse legacy .md for the single existing record stored there.
	if legacy == "" {
		if old, err := s.readFile(filepath.Join(s.Dir, ".md")); err == nil && old.Name == m.Name {
			filename = ".md"
		}
	}
	stage, err := s.stage(renderFrontmatter(m))
	if err != nil {
		return err
	}
	defer os.Remove(stage)
	path := filepath.Join(s.Dir, filename)
	if filename == unique {
		err = os.Rename(stage, path)
	} else if err = os.Link(stage, path); errors.Is(err, os.ErrExist) {
		old, readErr := s.readFile(path)
		if readErr != nil {
			return readErr
		}
		if old.Name != m.Name {
			path = filepath.Join(s.Dir, unique)
		}
		err = os.Rename(stage, path)
	}
	if err != nil {
		return err
	}
	return s.rebuildIndex()
}

// stage creates a complete file in the destination directory so publishing it
// with link or rename never exposes partial content.
func (s *Store) stage(content string) (string, error) {
	file, err := os.CreateTemp(s.Dir, ".memory-*")
	if err != nil {
		return "", err
	}
	_, writeErr := file.WriteString(content)
	if err := errors.Join(writeErr, file.Close()); err != nil {
		os.Remove(file.Name())
		return "", err
	}
	return file.Name(), nil
}

// Index renders the current memories, including records saved by other sessions.
func (s *Store) Index() string {
	mems, _ := s.Scan()
	return buildIndex(mems)
}

func (s *Store) rebuildIndex() error {
	mems, err := s.Scan()
	if err != nil {
		return err
	}
	stage, err := s.stage(buildIndex(mems))
	if err != nil {
		return err
	}
	defer os.Remove(stage)
	return os.Rename(stage, filepath.Join(s.Dir, indexFile))
}

func buildIndex(mems []Memory) string {
	var b strings.Builder
	b.WriteString("# Memory\n\n")
	if len(mems) == 0 {
		b.WriteString("(no memories yet)\n")
		return b.String()
	}
	for _, m := range mems {
		fmt.Fprintf(&b, "- [%s](%s) — %s\n", m.Name, filepath.Base(m.Path), m.Description)
	}
	return b.String()
}

// parseFrontmatter extracts name/description/type and the body from a markdown
// file with optional YAML frontmatter.
func parseFrontmatter(s string) Memory {
	var m Memory
	if strings.HasPrefix(s, "---") {
		rest := strings.TrimPrefix(s, "---")
		rest = strings.TrimLeft(rest, "\r\n")
		if i := strings.Index(rest, "\n---"); i >= 0 {
			head := rest[:i]
			body := rest[i+len("\n---"):]
			body = strings.TrimLeft(body, "-")
			m.Content = strings.TrimLeft(body, "\r\n")
			for _, line := range strings.Split(head, "\n") {
				k, v, ok := strings.Cut(line, ":")
				if !ok {
					continue
				}
				k = strings.TrimSpace(k)
				v = strings.TrimSpace(v)
				switch k {
				case "name":
					m.Name = v
				case "description":
					m.Description = v
				case "type":
					m.Type = Type(v)
				}
			}
			return m
		}
	}
	m.Content = s
	return m
}

func renderFrontmatter(m Memory) string {
	var b strings.Builder
	b.WriteString("---\n")
	fmt.Fprintf(&b, "name: %s\n", m.Name)
	fmt.Fprintf(&b, "description: %s\n", m.Description)
	if m.Type != "" {
		fmt.Fprintf(&b, "type: %s\n", m.Type)
	}
	b.WriteString("---\n\n")
	b.WriteString(strings.TrimRight(m.Content, "\n"))
	b.WriteString("\n")
	return b.String()
}
