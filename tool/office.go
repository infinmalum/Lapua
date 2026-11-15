package tool

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// Office documents (docx/xlsx/pptx) are OOXML: ZIP archives of XML parts. This
// file extracts their plain text with only the standard library. The
// extraction is structural — paragraphs, cell values, slide text runs — not
// styled: formatting, embedded media, and charts are not represented. That is
// enough for a model to read a document's content, and unlike native
// image/document blocks it works on every provider.

const (
	// Zip-bomb guards: every part is read through a per-entry cap and the whole
	// document through a total cap, so a crafted archive cannot expand
	// unbounded in memory.
	officeEntryCap = 32 << 20
	officeTotalCap = 64 << 20
)

// sniffOffice classifies a zip archive by its OOXML part prefixes: word/ →
// docx, xl/ → xlsx, ppt/ → pptx. ok=false means the archive is not an Office
// document, so Read falls through to its binary placeholder.
func sniffOffice(zr *zip.Reader) (kind string, ok bool) {
	var word, xl, ppt bool
	for _, f := range zr.File {
		switch {
		case strings.HasPrefix(f.Name, "word/"):
			word = true
		case strings.HasPrefix(f.Name, "xl/"):
			xl = true
		case strings.HasPrefix(f.Name, "ppt/"):
			ppt = true
		}
	}
	switch {
	case word:
		return "docx", true
	case xl:
		return "xlsx", true
	case ppt:
		return "pptx", true
	}
	return "", false
}

// officeReader bounds decompressed reads against the per-entry and total caps.
type officeReader struct {
	total int
}

// readPart returns the decompressed bytes of one zip entry, enforcing both
// caps.
func (r *officeReader) readPart(zr *zip.Reader, name string) ([]byte, error) {
	f, err := zr.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, officeEntryCap+1))
	if err != nil {
		return nil, err
	}
	if len(data) > officeEntryCap {
		return nil, fmt.Errorf("part %s exceeds %d decompressed bytes", name, officeEntryCap)
	}
	r.total += len(data)
	if r.total > officeTotalCap {
		return nil, fmt.Errorf("document exceeds %d decompressed bytes", officeTotalCap)
	}
	return data, nil
}

// extractOffice dispatches on the sniffed family and returns the joined plain
// text.
func extractOffice(zr *zip.Reader, kind string) (string, error) {
	or := &officeReader{}
	var text string
	var err error
	switch kind {
	case "docx":
		text, err = extractDocx(or, zr)
	case "xlsx":
		text, err = extractXlsx(or, zr)
	case "pptx":
		text, err = extractPptx(or, zr)
	default:
		return "", fmt.Errorf("unknown office kind %q", kind)
	}
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(text) == "" {
		return "(no text content found)", nil
	}
	return strings.TrimSpace(text), nil
}

// extractDocx pulls the paragraphs of word/document.xml.
func extractDocx(or *officeReader, zr *zip.Reader) (string, error) {
	data, err := or.readPart(zr, "word/document.xml")
	if err != nil {
		return "", err
	}
	return xmlParagraphText(data), nil
}

// extractPptx walks ppt/slides/slideN.xml in slide order (numeric, not
// lexicographic — slide10 sorts before slide2 as a string) with a header per
// slide.
func extractPptx(or *officeReader, zr *zip.Reader) (string, error) {
	type slide struct {
		n    int
		name string
	}
	var slides []slide
	for _, f := range zr.File {
		base := strings.TrimPrefix(f.Name, "ppt/slides/slide")
		if base == f.Name || !strings.HasSuffix(base, ".xml") {
			continue
		}
		if n, err := strconv.Atoi(strings.TrimSuffix(base, ".xml")); err == nil {
			slides = append(slides, slide{n, f.Name})
		}
	}
	if len(slides) == 0 {
		return "", fmt.Errorf("no slides found under ppt/slides/")
	}
	sort.Slice(slides, func(i, j int) bool { return slides[i].n < slides[j].n })
	var b strings.Builder
	for _, s := range slides {
		data, err := or.readPart(zr, s.name)
		if err != nil {
			return "", err
		}
		fmt.Fprintf(&b, "\n## Slide %d\n%s\n", s.n, strings.TrimSpace(xmlParagraphText(data)))
	}
	return strings.TrimSpace(b.String()), nil
}

// xmlParagraphText walks an OOXML drawing/wordprocessing part and joins its
// text runs: run text accumulates within a paragraph, one newline per
// paragraph, tab/br elements as line breaks. WordprocessingML (w:) and
// DrawingML (a:) use the same local names (t/p/tab/br), so one walker serves
// both docx and pptx. Malformed XML yields the text decoded so far rather than
// an error — partial content beats none.
func xmlParagraphText(data []byte) string {
	var b strings.Builder
	inText := false
	dec := xml.NewDecoder(bytes.NewReader(data))
	dec.Strict = false
	for {
		tok, err := dec.Token()
		if err != nil {
			break // EOF or malformed: keep what we have
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "t":
				inText = true
			case "tab":
				b.WriteByte('\t')
			case "br":
				b.WriteByte('\n')
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				inText = false
			case "p":
				b.WriteByte('\n')
			}
		case xml.CharData:
			if inText {
				b.Write(t)
			}
		}
	}
	return b.String()
}

// extractXlsx renders each worksheet as tab-separated rows. Sheet names come
// from xl/workbook.xml resolved through xl/_rels/workbook.xml.rels; string
// cells resolve through xl/sharedStrings.xml.
func extractXlsx(or *officeReader, zr *zip.Reader) (string, error) {
	shared := []string{}
	if data, err := or.readPart(zr, "xl/sharedStrings.xml"); err == nil {
		shared = readSharedStrings(data)
	}
	rels := map[string]string{}
	if data, err := or.readPart(zr, "xl/_rels/workbook.xml.rels"); err == nil {
		rels = readXlsxRels(data)
	}
	data, err := or.readPart(zr, "xl/workbook.xml")
	if err != nil {
		return "", err
	}
	sheets, err := xlsxSheetPaths(data, rels)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for _, sh := range sheets {
		sdata, err := or.readPart(zr, sh.path)
		if err != nil {
			return "", err
		}
		fmt.Fprintf(&b, "\n## %s\n", sh.name)
		writeSheetCells(&b, sdata, shared)
	}
	return strings.TrimSpace(b.String()), nil
}

type xlsxSheet struct {
	name string
	path string
}

// xlsxSheetPaths pairs workbook.xml's <sheet name=… r:id=…/> entries (in
// document order) with their worksheet part paths from the rels map.
func xlsxSheetPaths(data []byte, rels map[string]string) ([]xlsxSheet, error) {
	dec := xml.NewDecoder(bytes.NewReader(data))
	dec.Strict = false
	var out []xlsxSheet
	for {
		tok, err := dec.Token()
		if err != nil {
			break
		}
		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "sheet" {
			continue
		}
		name, rel := "", ""
		for _, a := range se.Attr {
			switch a.Name.Local {
			case "name":
				name = a.Value
			case "id": // r:id — the relationships namespace's id
				rel = a.Value
			}
		}
		sh := xlsxSheet{name: name}
		if t, ok := rels[rel]; ok {
			sh.path = "xl/" + strings.TrimPrefix(t, "/")
		}
		out = append(out, sh)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no sheets found in xl/workbook.xml")
	}
	return out, nil
}

// readXlsxRels maps relationship ids to targets (Type filtering is unnecessary:
// only worksheet targets are looked up).
func readXlsxRels(data []byte) map[string]string {
	dec := xml.NewDecoder(bytes.NewReader(data))
	dec.Strict = false
	out := map[string]string{}
	for {
		tok, err := dec.Token()
		if err != nil {
			break
		}
		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "Relationship" {
			continue
		}
		id, target := "", ""
		for _, a := range se.Attr {
			switch a.Name.Local {
			case "Id":
				id = a.Value
			case "Target":
				target = a.Value
			}
		}
		if id != "" && target != "" {
			out[id] = target
		}
	}
	return out
}

// readSharedStrings flattens xl/sharedStrings.xml's <si> items (each may hold
// several <t> rich-text runs) into one string per item.
func readSharedStrings(data []byte) []string {
	dec := xml.NewDecoder(bytes.NewReader(data))
	dec.Strict = false
	var out []string
	var cur strings.Builder
	inSI := false
	for {
		tok, err := dec.Token()
		if err != nil {
			break
		}
		switch t := tok.(type) {
		case xml.StartElement:
			if t.Name.Local == "si" {
				inSI = true
				cur.Reset()
			}
		case xml.EndElement:
			if t.Name.Local == "si" {
				inSI = false
				out = append(out, cur.String())
			}
		case xml.CharData:
			if inSI {
				cur.Write(t)
			}
		}
	}
	return out
}

// writeSheetCells renders one worksheet: rows in document order, cells
// tab-separated and gap-padded by their column letters so columns stay
// aligned. Shared-string cells (t="s") resolve through the shared table,
// inline-string cells carry their own <t> text, everything else passes the
// raw <v> text through.
func writeSheetCells(b *strings.Builder, data []byte, shared []string) {
	dec := xml.NewDecoder(bytes.NewReader(data))
	dec.Strict = false
	var rowCells []string
	lastCol := -1
	cellType, cellVal := "", strings.Builder{}
	cellCol := -1
	inValue := false
	flushCell := func() {
		val := cellVal.String()
		cellVal.Reset()
		if cellType == "s" {
			if idx, err := strconv.Atoi(val); err == nil && idx >= 0 && idx < len(shared) {
				val = shared[idx]
			}
		}
		for lastCol >= 0 && cellCol > lastCol+1 {
			rowCells = append(rowCells, "")
			lastCol++
		}
		rowCells = append(rowCells, val)
		lastCol = cellCol
		cellType = ""
		cellCol = -1
	}
	for {
		tok, err := dec.Token()
		if err != nil {
			break
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "row":
				rowCells, lastCol = nil, -1
			case "c":
				cellType, cellCol = "", -1
				for _, a := range t.Attr {
					switch a.Name.Local {
					case "t":
						cellType = a.Value
					case "r":
						cellCol = colIndex(a.Value)
					}
				}
			case "v", "t":
				inValue = true
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "v", "t":
				inValue = false
			case "c":
				flushCell()
			case "row":
				if len(rowCells) > 0 {
					b.WriteString(strings.Join(rowCells, "\t"))
					b.WriteByte('\n')
				}
			}
		case xml.CharData:
			if inValue {
				cellVal.Write(t)
			}
		}
	}
}

// colIndex converts an A1-style column reference's letters to a zero-based
// column index ("A"→0, "B"→1, "AA"→26).
func colIndex(ref string) int {
	n := 0
	for _, r := range ref {
		if r < 'A' || r > 'Z' {
			break
		}
		n = n*26 + int(r-'A') + 1
	}
	return n - 1
}
