# AANCA brand and design system

**Version:** 1.0  
**Date:** 1 September 2026  
**Scope:** visual identity, verbal identity, interfaces, scientific figures and public materials  
**Project:** AANCA — Automated Auditing of Nucleus Class Annotations

This document is the design authority for AANCA. It consolidates the identity already
implemented in the public article and turns it into reusable rules for websites,
presentations, reports, posters, figures, social materials and future interfaces.

It does not redefine the science. `SPEC.md`, frozen protocols and accepted evidence
remain authoritative for methods, results, completion stages and claim boundaries.
Whenever visual or marketing language conflicts with those sources, the scientific
source wins and publication stops until the conflict is resolved.

## 1. Brand core

### 1.1 What AANCA is

AANCA is a non-diagnostic university research prototype that ranks potentially
inconsistent nucleus class annotations for expert review. It does not determine
biological truth, diagnose disease or modify source annotations automatically.

The core interaction is a **second look**:

1. the source annotation remains visible and unchanged;
2. an identical visual copy is placed in a review queue;
3. evidence accompanies the recommendation;
4. a qualified expert decides what, if anything, happens next.

### 1.2 Brand promise

> Which annotations deserve a second look?

This question is the preferred editorial headline. It is curious and useful without
claiming that a flagged annotation is wrong.

### 1.3 Functional descriptor

Use this sentence when the audience needs a direct explanation:

> A group-safe framework that ranks potentially inconsistent annotations for expert
> review without changing source labels.

Polish supporting translation:

> Framework z bezpiecznym podziałem grupowym, który szereguje potencjalnie niespójne
> adnotacje do oceny eksperckiej bez zmieniania etykiet źródłowych.

The English wording is canonical for international scientific materials.

### 1.4 Personality

AANCA should feel:

- **precise** — exact values, explicit labels and visible provenance;
- **restrained** — one accent, quiet surfaces and no decorative noise;
- **candid** — negative, unavailable and adverse evidence remains visible;
- **human-centred** — the system recommends review and the expert retains agency;
- **research-led** — the visual hierarchy serves the question, method and evidence;
- **auditable** — important statements can be traced to a source.

AANCA should not feel clinical, omniscient, alarmist, cyberpunk, promotional or
autonomous.

### 1.5 Design principles

1. **Evidence before spectacle.** A graphic can clarify evidence but cannot decorate
   it into a stronger result.
2. **Recommendation, not verdict.** Every risk signal leads to expert review, never
   an automatic correction.
3. **Source remains intact.** Visual stories preserve the source object and move only
   a copy into the review queue.
4. **One visual system.** Dark editorial surfaces, neutral typography, violet focus
   and thin rules form one continuous language.
5. **Failure stays visible.** Missing, adverse and not-supported outcomes are labelled,
   not hidden or converted into zero.
6. **Motion is optional.** The complete message must survive reduced motion, script
   failure, print and small screens.

## 2. Naming and brand architecture

### 2.1 Official name

- Short name: **AANCA**
- Expanded name: **Automated Auditing of Nucleus Class Annotations**
- Website: **aancastudy.org**

Write `AANCA` in uppercase. Do not write `Aanca`, `AANCA.ai`, `AANCA AI` or expand the
acronym differently. Lowercase `aanca` is acceptable only in domains, package names,
file paths and machine identifiers.

At first mention in formal material use:

> AANCA (Automated Auditing of Nucleus Class Annotations)

After that, use `AANCA`.

### 2.2 Versions and studies

- **AANCA v1** identifies the frozen, auditable reference implementation.
- **AANCA V2** identifies the separate research programme and repository.
- A dataset or study is subordinate to the project, for example
  `AANCA / PanNuke controlled benchmark`.

Do not present V2 as a clinically validated successor or imply that a newer version
retroactively improves a frozen result.

### 2.3 Descriptor hierarchy

Use one of these levels, not several competing slogans in the same surface:

| Level | Preferred content | Use |
| --- | --- | --- |
| Brand | `AANCA` | navigation, cover, footer, avatar |
| Full name | `Automated Auditing of Nucleus Class Annotations` | first formal mention |
| Editorial question | `Which annotations deserve a second look?` | hero, talk title |
| Functional descriptor | group-safe ranking for expert review | abstract, README, metadata |
| Boundary | non-diagnostic; source annotations remain unchanged | footer, methods, review UI |

## 3. Verbal identity

### 3.1 Voice

Lead with the research question or supported outcome. Use short, concrete sentences.
Name the comparison, population and limitation. Prefer active voice and plain English.
Technical detail belongs near the evidence it qualifies.

The tone is confident about what was executed and deliberately cautious about what
the evidence cannot establish.

### 3.2 Required terminology

In English public material use these exact expressions:

- **potentially inconsistent annotation**;
- **recommended for expert review**;
- **source annotations remain unchanged**;
- **non-diagnostic research prototype**;
- **independent-expert disagreement**, when that is the measured endpoint;
- **controlled corruption** or **injected label change**, when that is the experiment.

Use `pre_corruption_label`, `observed_label`, `is_injected_corruption` and
`restored_label` as distinct technical terms. Do not collapse them into a generic
“correct/incorrect label” pair.

### 3.3 Prohibited claims

Do not write or imply:

- “AANCA detects pathology errors”;
- “the pathologist was wrong”;
- “AI-corrected labels”;
- “ground truth” when the source is an experimental reference or reviewer consensus;
- “clinically validated”, “diagnostic”, “safe for clinical use” or “patient benefit”;
- “automatic correction” or “self-healing dataset”;
- “independent validation” when only independent software recalculation occurred.

Disagreement is a review signal, not proof of error. Model disagreement is not a
medical verdict.

### 3.4 Evidence sentence pattern

Use this order:

> **Context → comparison → result → boundary.**

Example:

> In the controlled benchmark, the AANCA queue retrieved more injected label changes
> than an equal-budget random queue. This evaluates the injected process and does not
> establish naturally occurring annotation error.

### 3.5 Calls to action

Preferred:

- Inspect the evidence
- Reproduce the study
- Read the limitations
- Review the protocol
- Open the repository
- Recommended for expert review

Avoid:

- Find errors now
- Fix the labels
- Trust the AI
- Validate the diagnosis
- Improve your dataset automatically

### 3.6 Headline style

Use sentence case. Favour a clear question or factual statement. Do not use title case
for long headings, exclamation marks, fear-based language or unsupported superlatives.

Good:

- Which annotations deserve a second look?
- What the study actually learned
- Evidence at a glance
- Where the result stops

Avoid:

- Revolutionary AI Error Detection!
- Perfect Labels at Scale
- The Future of Digital Pathology

## 4. Logo system

### 4.1 Primary mark

The canonical AANCA mark is made from **four equal rounded square tiles** in a 2 × 2
grid, rotated as one group by 45 degrees.

Construction:

- tile side: `8u`;
- gap: `3u`;
- unrotated group: `19u × 19u`;
- corner radius: `2u` at the standard digital size;
- group rotation: exactly `45deg`;
- primary fill: AANCA Violet `#5E6AD2`.

The `8:3` tile-to-gap ratio is part of the identity. The mark contains four tiles;
do not add a centre or fifth tile. Historic micro-icons with another tile count are
not separate AANCA marks and should not be propagated into new assets.

Canonical SVG construction:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="AANCA">
  <g fill="#5E6AD2" transform="rotate(45 32 32)">
    <rect x="13" y="13" width="16" height="16" rx="4" />
    <rect x="35" y="13" width="16" height="16" rx="4" />
    <rect x="13" y="35" width="16" height="16" rx="4" />
    <rect x="35" y="35" width="16" height="16" rx="4" />
  </g>
</svg>
```

### 4.2 Lockup

The preferred lockup is the mark followed by the live-text wordmark `AANCA`.

- wordmark typeface: Inter;
- wordmark weight: 600;
- wordmark tracking: `-0.02em`;
- gap at the standard navigation size: `9px`;
- standard wordmark size: `14px`;
- wordmark colour on dark: Ink `#F7F8F8`.

Use live text in interfaces. In final artwork the wordmark may be converted to
outlines, but retain an editable source version.

### 4.3 Variants

| Variant | Mark | Wordmark | Background |
| --- | --- | --- | --- |
| Primary dark | `#5E6AD2` | `#F7F8F8` | `#010102` |
| Interactive dark | `#828FFF` | `#F7F8F8` | `#010102` |
| Print/light | `#4B56BB` | `#111111` | white |
| One-colour dark | white | white | black/dark image |
| One-colour light | black | black | white/light image |

Never place the violet mark on a background that makes its silhouette or contrast
unclear. When an image is unavoidable, use a Canvas-coloured field behind the full
lockup.

### 4.4 Clear space and minimum size

Let `X` equal one tile side.

- keep at least `1X` clear space around the visual bounds of the mark;
- keep at least `1X` above and below the full lockup;
- do not align other diamonds, cells or chart points inside this zone;
- minimum digital construction size: `19px × 19px` before rotation;
- minimum wordmark size: `14px`;
- minimum print mark: `5mm` before rotation.

If the lockup becomes unclear, use the mark alone with an accessible text label.

### 4.5 Logo misuse

Do not:

- stretch, skew or rotate the group by an angle other than 45 degrees;
- change the tile count, ratio, spacing or relative size;
- recolour individual tiles as class labels;
- use gradients, glows, shadows, bevels or 3D effects;
- place the mark inside a medical cross, shield, brain or microscope symbol;
- use red to make the mark look like an error alarm;
- animate the wordmark as part of the nucleus-to-mark transformation;
- use the mark as a data point that could be mistaken for a scientific result.

## 5. Colour system

### 5.1 Dark interface palette

| Token | Value | Role |
| --- | --- | --- |
| Canvas | `#010102` | page background, hero, dominant field |
| Deep surface | `#07080A` | rare high-emphasis evidence surface |
| Surface 1 | `#0F1011` | navigation, menu, quiet containers |
| Surface 2 | `#141516` | inputs, secondary panels |
| Ink | `#F7F8F8` | headings, key values, primary copy |
| Ink muted | `#D0D6E0` | body copy, supporting statements |
| Ink subtle | `#8A8F98` | captions, metadata, secondary labels |
| Ink tertiary | `#7B8089` | low-emphasis labels and axes |
| Line | `#23252A` | standard hairlines |
| Line strong | `#34343A` | active boundaries and controls |
| AANCA Violet | `#5E6AD2` | mark, non-text emphasis, data points |
| Focus Violet | `#828FFF` | links, focus, interactive and text accent |

The page should remain mostly Canvas and neutral ink. Violet is a precise signal, not
a wash. As a practical composition target, keep violet below roughly 5% of the visible
area.

### 5.2 Light and print palette

| Token | Value |
| --- | --- |
| Canvas | `#FFFFFF` |
| Surface | `#F6F6F6` |
| Ink | `#111111` |
| Ink muted | `#333333` |
| Ink subtle | `#555555` |
| Ink tertiary | `#666666` |
| Line | `#CCCCCC` |
| Line strong | `#999999` |
| Print Violet | `#4B56BB` |

Print layouts reverse to a white ground. Do not print large areas of near-black unless
the format specifically requires a dark cover.

### 5.3 Contrast rules

Measured WCAG contrast on Canvas:

| Foreground on `#010102` | Contrast | Use |
| --- | ---: | --- |
| `#F7F8F8` | 19.61:1 | all text |
| `#D0D6E0` | 14.28:1 | all text |
| `#8A8F98` | 6.42:1 | normal secondary text |
| `#7B8089` | 5.26:1 | normal tertiary text |
| `#5E6AD2` | 4.44:1 | shapes and large text only |
| `#828FFF` | 7.27:1 | normal accent text and focus |

Base Violet is just below 4.5:1 on Canvas and falls further on raised surfaces. Do not
use it for small normal text. Use Focus Violet for accent copy, links and keyboard
focus.

### 5.4 Semantic colour

AANCA does not use green as a shortcut for “scientifically true” or red as a shortcut
for “wrong annotation”. Supported, adverse, unavailable and not-supported results use
explicit text, shape and line style. Colour can reinforce a state only when a text
label carries the meaning.

## 6. Typography

### 6.1 Families

Primary sans-serif:

```css
font-family: Inter, "SF Pro Display", -apple-system, BlinkMacSystemFont,
  "Segoe UI", sans-serif;
```

Data and provenance mono:

```css
font-family: "JetBrains Mono", "Cascadia Mono", "SFMono-Regular",
  Consolas, monospace;
```

Use Inter for editorial copy and interface labels. Use JetBrains Mono only for hashes,
machine IDs, exact metrics, short labels and commands. Do not set paragraphs in mono.

### 6.2 Weight

- 400: body and long-form reading;
- 500: section headings, key statements and numeric emphasis;
- 600: hero, brand wordmark, small strong labels and controls.

Avoid weights below 400 or above 600. The identity relies on clarity, not heavy display
type.

### 6.3 Type scale

| Style | Size | Line height | Tracking |
| --- | --- | --- | --- |
| Hero | `clamp(34px, 5vw, 52px)` | 1.08 | `-0.04em` |
| Mobile hero | `clamp(32px, 11vw, 42px)` | 1.08 | `-0.04em` |
| Display section | `clamp(25px, 3vw, 38px)` | 1.08 | `-0.035em` |
| Editorial lead | `clamp(22px, 2.15vw, 29px)` | 1.48 | `-0.03em` |
| Hero lead | `clamp(16px, 1.35vw, 19px)` | 1.62 | `-0.01em` |
| Body | `16px` | 1.74 | `-0.008em` |
| UI | `12–14px` | 1.45–1.6 | `0` to `0.01em` |
| Mono label | `10px` | 1.4 | `0.07–0.10em` |

Use sentence case. Uppercase is reserved for short mono labels, hypotheses, status
codes and compact metadata. Never set long headings or paragraphs in uppercase.

### 6.4 Line length and alignment

- long-form editorial rail: maximum `640px`;
- hero copy: maximum `620px`;
- explanatory headings may use balanced wrapping;
- paragraphs are left aligned;
- centred text is reserved for short evidence overviews, not long reading;
- numeric columns use tabular figures and right alignment where comparison benefits.

## 7. Layout and spacing

### 7.1 Width rails

| Rail | Width | Purpose |
| --- | ---: | --- |
| Editorial | `640px` | prose, protocols, interpretation |
| Figure | `1080px` | charts, evidence tables, media |
| Wide | `1160px` | navigation, footer, multi-column evidence |
| Hero | `1320px` | left-aligned copy with right-side review field |

Page padding is `clamp(22px, 4.5vw, 72px)`. A nested component inside a padded rail
must not subtract page padding again.

### 7.2 Spacing scale

| Token | Value |
| --- | ---: |
| XS | `8px` |
| SM | `12px` |
| MD | `16px` |
| LG | `24px` |
| XL | `32px` |
| 2XL | `48px` |
| Paragraph gap | `1.35em` |
| Block gap | `28px` |
| Caption gap | `10px` |
| Section space | `clamp(52px, 5.5vw, 76px)` |

Prefer open space and one clear relationship over stacked panels. Separate top-level
sections with one hairline. Do not add multiple decorative rules.

### 7.3 Grid behaviour

- desktop evidence grids may use two or four columns;
- collapse complex figures before labels or values become cramped;
- below `1040px`, simplify multi-column result layouts;
- below `901px`, remove sticky findings and keep every answer in document flow;
- below `720px`, use one-column navigation, evidence, rules and footer;
- never introduce horizontal page scrolling.

### 7.4 Radius and depth

| Element | Radius |
| --- | ---: |
| Mark tile | `2px` at standard size |
| Chips / compact badges | `4px` |
| Controls / buttons / media | `8px` |
| Large disclosure / mobile menu | `12px` |

Use borders and surface tone instead of box shadows. The only atmospheric depth in
the core system is the fixed navigation backdrop blur.

## 8. Interface components

### 8.1 Navigation

- fixed height: `60px`;
- Canvas at approximately 82% opacity;
- `20px` backdrop blur with restrained saturation;
- lockup left, short section links right;
- link text: 12px/500 in Ink subtle;
- hover and active: Focus Violet;
- mobile control: at least `44 × 44px` with visible expanded state.

### 8.2 Buttons and links

Primary actions are text links or quiet bordered buttons. Use `44px` minimum target
height. A button has Surface 2 fill, Line strong border, 8px radius and Ink text.

Do not use large filled violet marketing buttons as the default. Violet should mark
focus and evidence, not simulate urgency.

### 8.3 Cards and disclosures

Most information should live directly on the editorial surface. Use a card only when
the boundary has meaning, such as a seed identity, checksum authority or compact result
summary.

- standard cards: transparent or Surface 1/2, 1px line;
- strong evidence disclosure: 12px radius and restrained dark gradient;
- never nest cards more than one level;
- labels stay visible when the disclosure is closed;
- adverse and unavailable states may not be hidden by default if they qualify a claim.

### 8.4 Forms and filters

- minimum control height: `44px`;
- Surface 2 background, Line strong border, 8px radius;
- persistent visible label; placeholder is not a label;
- keyboard focus: 2px Focus Violet outline with 4px offset;
- tables must become labelled record blocks on narrow screens rather than overflow.

### 8.5 Status labels

Use the exact scientific stage vocabulary from `SPEC.md`. Display machine states in
mono or compact uppercase, but explain them in ordinary language nearby. Never invent
an intermediate completion stage for visual convenience.

## 9. Data visualisation

### 9.1 Visual grammar

- neutral axis and grid: Line / Line strong;
- primary point: Violet diamond, typically a rotated rounded square;
- interactive or endpoint point: Focus Violet;
- confidence interval: 2px line with clear endpoints;
- exact numeric value: JetBrains Mono with tabular figures;
- comparison direction and zero/reference line must be explicit;
- text labels carry meaning; colour alone does not.

The diamond data marker echoes the mark but must remain smaller and clearly embedded
in a labelled quantitative axis.

### 9.2 Evidence rules

1. Read values from machine evidence; never type or round them by eye.
2. Preserve adverse, neutral, failed and unavailable outcomes.
3. Plot `unavailable` as unavailable, never at zero.
4. Show confidence intervals and sample/group support where they affect interpretation.
5. Keep controlled corruption, natural disagreement and downstream utility visually
   and verbally separate.
6. Do not pool datasets or endpoints merely to create a larger headline number.
7. State whether a value is confirmatory, exploratory, descriptive or a saved-evidence
   readback.

### 9.3 Chart style

Prefer forest plots, dot-and-interval plots, restrained bars for counts and simple
tables. Avoid 3D charts, gauges, traffic-light scorecards, pie charts with many slices,
glowing heatmaps and decorative medical dashboards.

One chart should answer one question. Supporting explanation belongs directly above
or below it.

## 10. Imagery and scientific media

### 10.1 Signature visual language

The AANCA visual field uses isolated nucleus forms on Canvas, quiet blue-violet tissue
detail and thin violet review frames. The field should feel like careful inspection,
not a diagnostic scanner.

Preferred composition:

- black or near-black ground;
- sparse, irregular biological forms with ample negative space;
- consistent violet target indication independent of class;
- corner frames or contours rather than aggressive red boxes;
- source object retained when a copy moves into a review queue.

### 10.2 Scientific images

- preserve aspect ratio and pixel integrity;
- cite dataset, fold/subset and transformation;
- distinguish raw image, derived crop, overlay and decorative rendering;
- do not retouch evidence to look more convincing;
- do not encode class information into target highlighting used by the model;
- overlays must say what was changed visually and what remained source data.

### 10.3 Generated and decorative images

AI-generated or artist-created media may be used only as clearly decorative brand
material. It must never be presented as a real specimen, dataset sample, experiment
output or validation evidence. Scientific figures and example nuclei require traceable
source provenance.

### 10.4 Iconography

Use simple 1–1.5px line icons with square or softly rounded geometry. Prefer literal
concepts such as inspect, evidence, group split, queue and reproduce. Avoid medical
crosses, robotic brains, magic wands, warning sirens and generic “AI sparkle” marks.

## 11. Motion

### 11.1 Motion principles

Motion explains selection, copying, grouping and review. It never implies that AANCA
changes the source annotation.

- source nuclei remain in place;
- visual copies travel into the review queue;
- camera, grouping and colour can reveal the mark;
- easing is continuous and calm;
- no flashing, pulsing error state or celebratory success animation;
- interface transitions usually last `160–500ms`.

### 11.2 Signature hero sequence

The current Second-Look Review Field is the reference brand motion:

1. an audit frame deliberately visits four desktop or two mobile nuclei;
2. identical copies enter a short expert-review queue;
3. one patch enlarges and separates into four square patches;
4. camera pull-out, sibling separation, 45° rotation and change to Violet overlap;
5. the completed four-tile mark holds for 1.8 seconds;
6. the mark spins and passes through the camera into a full-black transition;
7. the unchanged field returns in eight staggered groups.

The animation is decorative, capped at 60 rendered frames per second and suspended
when hidden. Do not add a Canvas wordmark to the transformation.

### 11.3 Reduced motion

Under `prefers-reduced-motion: reduce`:

- stop looping motion;
- show a meaningful static final frame;
- expose all findings in normal document flow;
- disable smooth scrolling and non-essential transitions;
- never withhold content because its animation did not run.

## 12. Accessibility

Every AANCA surface must:

- meet WCAG AA contrast for text and meaningful controls;
- use Focus Violet for keyboard focus;
- provide a skip link on long pages;
- use semantic headings in document order;
- expose an accessible name for a mark-only link;
- give informative images useful alt text and decorative images empty alt text;
- preserve all information without colour, hover or motion;
- use at least `44 × 44px` touch targets for primary mobile controls;
- support zoom and reflow without horizontal scrolling;
- keep scientific tables understandable when transformed into mobile records;
- provide a static or print-safe alternative for animated explanations.

Accessibility is part of the evidence boundary: if a limitation or adverse result is
not perceivable, the presentation is incomplete.

## 13. Channel applications

### 13.1 Website and product UI

Use the dark system by default. Recommended order:

1. research question;
2. concise non-diagnostic descriptor;
3. evidence at a glance;
4. method;
5. findings, including adverse evidence;
6. provenance and reproduction;
7. current stage and limitations;
8. author and terms.

Avoid a conventional SaaS landing page with testimonials, pricing cards, fabricated
users or conversion-first claims.

### 13.2 Presentations

- 16:9 Canvas background;
- mark and wordmark on cover and closing slide;
- one research question or result per slide;
- use the same type, colour and chart grammar as the website;
- keep citations and evidence identity visible but secondary;
- include a scope/limitations slide before the final conclusion;
- do not turn a null or adverse result into a green “success” slide.

### 13.3 Reports and papers

Use the light/print palette for long printed reading. The first page should contain the
full name, short descriptor, author, date/version and non-diagnostic boundary. Figures
must remain legible in grayscale and carry self-contained captions.

### 13.4 Posters

Use a dark cover band or full dark canvas only when print quality supports it. Preserve
the editorial hierarchy: question, method, evidence, boundary. Do not fill the poster
with equal-weight cards.

### 13.5 Repository and technical documentation

Lead with the full name and one-sentence purpose. Use Markdown tables only when they
improve scanning. Commands, hashes and identifiers use mono. Current evidence and
stage text must be produced or verified against authoritative artifacts.

### 13.6 Social and thumbnails

Use the mark, a short question and one restrained nucleus field. Do not place an
unsupported metric without its endpoint or turn a review recommendation into an
“error found” badge. Keep critical copy inside the central 80% safe area.

## 14. Implementation tokens

The canonical reusable CSS foundation is:

```css
:root {
  color-scheme: dark;
  --aanca-canvas: #010102;
  --aanca-surface-1: #0f1011;
  --aanca-surface-2: #141516;
  --aanca-ink: #f7f8f8;
  --aanca-ink-muted: #d0d6e0;
  --aanca-ink-subtle: #8a8f98;
  --aanca-ink-tertiary: #7b8089;
  --aanca-line: #23252a;
  --aanca-line-strong: #34343a;
  --aanca-accent: #5e6ad2;
  --aanca-accent-focus: #828fff;

  --aanca-sans: Inter, "SF Pro Display", -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  --aanca-mono: "JetBrains Mono", "Cascadia Mono", "SFMono-Regular",
    Consolas, monospace;

  --aanca-editorial: 640px;
  --aanca-figure: 1080px;
  --aanca-wide: 1160px;
  --aanca-hero: 1320px;
  --aanca-page-pad: clamp(22px, 4.5vw, 72px);

  --aanca-space-xs: 8px;
  --aanca-space-sm: 12px;
  --aanca-space-md: 16px;
  --aanca-space-lg: 24px;
  --aanca-space-xl: 32px;
  --aanca-space-2xl: 48px;
  --aanca-section-space: clamp(52px, 5.5vw, 76px);
}
```

Reference lockup:

```html
<a class="aanca-brand" href="/" aria-label="AANCA — home">
  <span class="aanca-mark" aria-hidden="true">
    <i></i><i></i><i></i><i></i>
  </span>
  <span>AANCA</span>
</a>
```

```css
.aanca-brand {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  color: var(--aanca-ink);
  font: 600 14px/1 var(--aanca-sans);
  letter-spacing: -0.02em;
  text-decoration: none;
}

.aanca-mark {
  width: 19px;
  height: 19px;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 3px;
  transform: rotate(45deg);
}

.aanca-mark i {
  display: block;
  border-radius: 2px;
  background: var(--aanca-accent);
}
```

## 15. Reusable copy blocks

### 15.1 Short English description

> AANCA is a non-diagnostic research prototype that prioritises potentially
> inconsistent nucleus class annotations for expert review without changing source
> annotations automatically.

### 15.2 Short Polish description

> AANCA jest niediagnostycznym prototypem badawczym, który priorytetyzuje potencjalnie
> niespójne adnotacje klas jąder komórkowych do oceny eksperckiej i nie zmienia
> automatycznie adnotacji źródłowych.

### 15.3 Review-queue label

> Recommended for expert review

Not:

> Error detected

### 15.4 Standard boundary

> A high audit score indicates model-based inconsistency evidence. It does not prove
> that the source annotation or a pathologist is wrong.

### 15.5 Footer summary

> Automated auditing of nucleus class annotations: a university research prototype
> for prioritising potentially inconsistent annotations for expert review.
> Non-diagnostic research; source annotations are never changed automatically.

## 16. Release checklist

Before approving an AANCA asset, confirm:

### Identity

- [ ] `AANCA` is uppercase and the expanded name is correct.
- [ ] The mark has exactly four tiles, the 8:3 ratio and a 45° group rotation.
- [ ] Colours and typography use the defined tokens.
- [ ] Violet is a signal, not the dominant field.

### Scientific language

- [ ] The item says “potentially inconsistent annotation” and “recommended for
      expert review” where applicable.
- [ ] It does not claim diagnosis, pathologist error or automatic correction.
- [ ] Controlled corruption, natural disagreement and downstream utility are separate.
- [ ] Adverse, unavailable and failed results remain visible.
- [ ] Completion stages match `SPEC.md` and authoritative evidence.

### Visual evidence

- [ ] Every metric is sourced, labelled and paired with its endpoint.
- [ ] Confidence intervals and support are shown when required.
- [ ] Unavailable values are not plotted at zero.
- [ ] Scientific images have provenance and transformations are disclosed.
- [ ] Decorative/generated imagery cannot be mistaken for evidence.

### Accessibility and delivery

- [ ] Text contrast, focus and touch targets pass.
- [ ] Content works without motion, hover and colour.
- [ ] Mobile has no horizontal overflow.
- [ ] Print/static alternatives preserve the full message.
- [ ] Links, assets, alt text and console output are verified.

## 17. Governance

This file controls design and brand consistency only. It does not change an AANCA
model, annotation, dataset, metric, evidence artifact or scientific completion stage.

When the identity changes materially:

1. record the binding decision in `DECISIONS.md`;
2. update this file and the implementation tokens together;
3. regenerate affected public artifacts rather than editing generated output;
4. run accessibility, responsive, print and checksum/package checks;
5. record executed validation in `STATUS.md`.

Scientific claims must continue to be generated or verified against saved evidence.
A visually stronger treatment never authorises a stronger conclusion.

## 18. Existing implementation references

- Public presentation styles: `src/histo_audit/assets/mvp_presentation.css`
- Signature motion: `src/histo_audit/assets/hero-review-field.js`
- Generated public article: `src/histo_audit/mvp_demo.py`
- Presentation hero reference: `docs/assets/aanca-presentation-hero.png`
- Scientific language and scope: `SPEC.md`
- Ethics and prohibited claims: `ETHICS_AND_LIMITATIONS.md`
- Binding presentation decisions: `DECISIONS.md` (especially D033 and D038–D041)

