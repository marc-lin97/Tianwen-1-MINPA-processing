/* Build the Word version of the MINPA spectrum/moments algorithm note.
 *
 * Source of truth: docs/tw1_minpa_spectrum_moments_algorithm.md
 * Usage (PowerShell):
 *   $env:NODE_PATH='<bundled node_modules>'
 *   node scripts/build_tw1_minpa_algorithm_docx.js
 */

const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  LevelFormat,
  Packer,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const repoRoot = path.resolve(__dirname, "..");
const mdPath = path.join(repoRoot, "docs", "tw1_minpa_spectrum_moments_algorithm.md");
const outPath = path.join(repoRoot, "docs", "tw1_minpa_spectrum_moments_algorithm.docx");

const COLORS = {
  navy: "17365D",
  blue: "2F75B5",
  paleBlue: "D9EAF7",
  paleGray: "F2F2F2",
  midGray: "D9E1F2",
  text: "1F1F1F",
  white: "FFFFFF",
};

function inlineRuns(text, options = {}) {
  const runs = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  for (const match of text.matchAll(re)) {
    if (match.index > last) {
      runs.push(new TextRun({ text: text.slice(last, match.index), ...options }));
    }
    const token = match[0];
    if (token.startsWith("`")) {
      runs.push(new TextRun({
        text: token.slice(1, -1),
        font: "Consolas",
        color: "7030A0",
        ...options,
      }));
    } else {
      runs.push(new TextRun({ text: token.slice(2, -2), bold: true, ...options }));
    }
    last = match.index + token.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), ...options }));
  return runs.length ? runs : [new TextRun({ text, ...options })];
}

function paragraph(text, opts = {}) {
  return new Paragraph({
    children: inlineRuns(text),
    spacing: { after: opts.after ?? 120, line: 340 },
    alignment: opts.alignment,
    indent: opts.indent,
    shading: opts.shading,
    border: opts.border,
    keepNext: opts.keepNext,
  });
}

function codeBlock(lines) {
  return new Paragraph({
    children: [new TextRun({ text: lines.join("\n"), font: "Consolas", size: 18, color: "333333" })],
    spacing: { before: 80, after: 140, line: 270 },
    indent: { left: 280, right: 180 },
    shading: { type: ShadingType.CLEAR, fill: "F7F7F7", color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: COLORS.blue } },
  });
}

function quote(text) {
  return new Paragraph({
    children: inlineRuns(text, { color: "294E70" }),
    spacing: { before: 100, after: 150, line: 330 },
    indent: { left: 300, right: 200 },
    shading: { type: ShadingType.CLEAR, fill: "EAF3F8", color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: COLORS.blue } },
  });
}

function tableFromRows(rows) {
  const colCount = Math.max(...rows.map((r) => r.length));
  const pageWidth = 9354;
  const colWidth = Math.floor(pageWidth / colCount);
  const borders = {
    top: { style: BorderStyle.SINGLE, size: 4, color: "B7C9D6" },
    bottom: { style: BorderStyle.SINGLE, size: 4, color: "B7C9D6" },
    left: { style: BorderStyle.SINGLE, size: 4, color: "B7C9D6" },
    right: { style: BorderStyle.SINGLE, size: 4, color: "B7C9D6" },
    insideHorizontal: { style: BorderStyle.SINGLE, size: 3, color: "D9E2F3" },
    insideVertical: { style: BorderStyle.SINGLE, size: 3, color: "D9E2F3" },
  };
  return new Table({
    width: { size: pageWidth, type: WidthType.DXA },
    columnWidths: Array(colCount).fill(colWidth),
    rows: rows.map((cells, rowIndex) => new TableRow({
      tableHeader: rowIndex === 0,
      cantSplit: true,
      children: Array.from({ length: colCount }, (_, i) => new TableCell({
        width: { size: colWidth, type: WidthType.DXA },
        shading: rowIndex === 0
          ? { type: ShadingType.CLEAR, fill: COLORS.navy, color: "auto" }
          : rowIndex % 2 === 0
            ? { type: ShadingType.CLEAR, fill: "F7FAFC", color: "auto" }
            : undefined,
        margins: { top: 80, bottom: 80, left: 90, right: 90 },
        children: [new Paragraph({
          children: inlineRuns(cells[i] || "", {
            color: rowIndex === 0 ? COLORS.white : COLORS.text,
            bold: rowIndex === 0,
            size: 18,
          }),
          spacing: { after: 0, line: 260 },
        })],
      })),
    })),
    borders,
  });
}

function stripYaml(lines) {
  if (lines[0] !== "---") return lines;
  const end = lines.indexOf("---", 1);
  return end >= 0 ? lines.slice(end + 1) : lines;
}

function parseMarkdown(md) {
  const lines = stripYaml(md.replace(/\r\n/g, "\n").split("\n"));
  const content = [];
  let i = 0;
  let firstH1 = true;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim() || line.trim() === "---") { i += 1; continue; }
    if (line.startsWith("```")) {
      const block = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) block.push(lines[i++]);
      i += 1;
      content.push(codeBlock(block));
      continue;
    }
    if (line.startsWith("|")) {
      const tableLines = [];
      while (i < lines.length && lines[i].startsWith("|")) tableLines.push(lines[i++]);
      const rows = tableLines
        .map((s) => s.split("|").slice(1, -1).map((x) => x.trim()))
        .filter((r) => !r.every((x) => /^:?-+:?$/.test(x)));
      content.push(tableFromRows(rows));
      content.push(new Paragraph({ spacing: { after: 100 } }));
      continue;
    }
    if (line.startsWith("# ")) {
      if (firstH1) { firstH1 = false; i += 1; continue; }
      content.push(new Paragraph({ text: line.slice(2), heading: HeadingLevel.TITLE, pageBreakBefore: true }));
      i += 1;
      continue;
    }
    if (line.startsWith("## ")) {
      content.push(new Paragraph({ text: line.slice(3), heading: HeadingLevel.HEADING_1, pageBreakBefore: /^\d+\. /.test(line.slice(3)) && /^(5|7|11|15)\./.test(line.slice(3)) }));
      i += 1;
      continue;
    }
    if (line.startsWith("### ")) {
      content.push(new Paragraph({ text: line.slice(4), heading: HeadingLevel.HEADING_2 }));
      i += 1;
      continue;
    }
    if (line.startsWith("> ")) {
      content.push(quote(line.slice(2)));
      i += 1;
      continue;
    }
    const bullet = line.match(/^[-*] (.+)$/);
    if (bullet) {
      content.push(new Paragraph({
        children: inlineRuns(bullet[1]),
        numbering: { reference: "bullet-list", level: 0 },
        spacing: { after: 70, line: 320 },
      }));
      i += 1;
      continue;
    }
    const numbered = line.match(/^(\d+)\. (.+)$/);
    if (numbered) {
      content.push(new Paragraph({
        children: inlineRuns(numbered[2]),
        numbering: { reference: "number-list", level: 0 },
        spacing: { after: 70, line: 320 },
      }));
      i += 1;
      continue;
    }

    const paraLines = [line.trim()];
    i += 1;
    while (i < lines.length && lines[i].trim() && !/^(#|>|```|\||[-*] |\d+\. )/.test(lines[i])) {
      paraLines.push(lines[i].trim());
      i += 1;
    }
    content.push(paragraph(paraLines.join(" ")));
  }
  return content;
}

const markdown = fs.readFileSync(mdPath, "utf8");
const body = parseMarkdown(markdown);

const titlePage = [
  new Paragraph({ spacing: { before: 1600 } }),
  new Paragraph({
    children: [new TextRun({ text: "天问一号 MINPA", bold: true, size: 44, color: COLORS.navy, font: "Microsoft YaHei" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 180 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "从初始粒子数据到能谱与各阶矩的算法说明", bold: true, size: 36, color: COLORS.blue, font: "Microsoft YaHei" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 420 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "当前项目方法总结 · 版本 1.0", size: 24, color: "666666", font: "Microsoft YaHei" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "更新日期：2026-07-15", size: 21, color: "666666", font: "Microsoft YaHei" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 900 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "范围：H+、O+、O2+；全能级与 >1 keV 支路；MINPA/星体、MSO、MSE 坐标", size: 20, color: COLORS.navy, font: "Microsoft YaHei" })],
    alignment: AlignmentType.CENTER,
    shading: { type: ShadingType.CLEAR, fill: COLORS.paleBlue, color: "auto" },
    spacing: { before: 180, after: 180, line: 320 },
    indent: { left: 520, right: 520 },
  }),
  new Paragraph({ pageBreakBefore: true, text: "目录", heading: HeadingLevel.HEADING_1 }),
  paragraph("0. 给 ChatGPT 的读取约定　　1. 当前算法的总体结构　　2. 输入数据与来源追踪"),
  paragraph("3. 模式、记录拆分和数据立方体　　4. 能量、质量和角度几何　　5. 从 DPF 得到三物种能谱"),
  paragraph("6. 速度单元和矩积分权重　　7. 各阶矩的定义和当前实现状态　　8. 能量宽度的两种当前口径"),
  paragraph("9. 从仪器坐标到 MSO 和 MSE　　10. 质量标志、有效性和状态码　　11. 最终产品怎样形成"),
  paragraph("12. 可复现伪代码　　13. 强制科学检查　　14. 已知限制与后续工作　　15. 源代码与方法文件索引　　16. 变量速查"),
  new Paragraph({ pageBreakBefore: true }),
];

const doc = new Document({
  creator: "Codex",
  title: "天问一号 MINPA 从初始粒子数据到能谱与各阶矩的算法说明",
  subject: "Tianwen-1 MINPA particle spectra and moments algorithm",
  description: "Current project algorithm summary generated from the repository method notes and code.",
  styles: {
    default: {
      document: { run: { font: "Microsoft YaHei", size: 21, color: COLORS.text }, paragraph: { spacing: { line: 340 } } },
      heading1: { run: { font: "Microsoft YaHei", size: 30, bold: true, color: COLORS.navy }, paragraph: { spacing: { before: 260, after: 130 }, keepNext: true } },
      heading2: { run: { font: "Microsoft YaHei", size: 25, bold: true, color: COLORS.blue }, paragraph: { spacing: { before: 210, after: 100 }, keepNext: true } },
      title: { run: { font: "Microsoft YaHei", size: 36, bold: true, color: COLORS.navy }, paragraph: { spacing: { after: 180 }, keepNext: true } },
    },
  },
  numbering: {
    config: [
      { reference: "bullet-list", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
      { reference: "number-list", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1134, right: 1276, bottom: 1134, left: 1276, header: 560, footer: 560 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        children: [new TextRun({ text: "Tianwen-1 MINPA · 能谱与矩算法", size: 17, color: "6D7F8F", font: "Microsoft YaHei" })],
        alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "B7C9D6" } },
        spacing: { after: 50 },
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        children: [new TextRun({ text: "当前项目方法总结  |  ", size: 17, color: "777777" }), new TextRun({ children: [PageNumber.CURRENT], size: 17, color: "777777" })],
        alignment: AlignmentType.CENTER,
      })] }),
    },
    children: [...titlePage, ...body],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outPath, buffer);
  process.stdout.write(`Wrote ${outPath}\n`);
});
