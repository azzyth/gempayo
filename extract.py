import pdfplumber, re, csv, bisect
from collections import defaultdict

PATH = "katalog_bmkg.pdf"
FIRST_PAGE, LAST_PAGE = 151, 350      # 2023 -> 2000
LOOKAHEAD = 351                        # provides end boundary for the last record

HEADER_LABELS = {
    "No.", "No", "Tanggal/Wilayah", "Origin", "Time", "Koordinat", "(0)",
    "Lat", "Long", "Depth", "(km)", "Mag", "Wilayah", "yang", "merasakan",
    "Korban/Kerusakan", "Korban", "/", "Kerusakan", "Sumber",
}
FOOTER_TOP = 525
TOL = 4.0

MONTHS = ("Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|"
          "Oktober|November|Desember")
DATE_FULL = re.compile(rf"(\d{{1,2}})\s+({MONTHS})\s+(\d{{4}})")
DATE_LINE = re.compile(rf"^(\d{{1,2}})\s+({MONTHS})\b")
DATE_ABBR = re.compile(r"^(\d{1,2})-([A-Za-z]{3})-(\d{2})$")
ABBR_MONTH = {"Jan": "Januari", "Feb": "Februari", "Mar": "Maret", "Apr": "April",
              "Mei": "Mei", "Jun": "Juni", "Jul": "Juli", "Agu": "Agustus",
              "Aug": "Agustus", "Sep": "September", "Okt": "Oktober",
              "Oct": "Oktober", "Nov": "November", "Des": "Desember",
              "Dec": "Desember"}


def merge_close(vals, tol=12.0):
    out = []
    for v in sorted(vals):
        if out and v - out[-1] <= tol:
            out[-1] = (out[-1] + v) / 2
        else:
            out.append(v)
    return out


def main_table_columns(page):
    tbls = [t for t in page.find_tables() if (t.bbox[2] - t.bbox[0]) > 500]
    if not tbls:
        return None, None
    t = max(tbls, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    bounds = merge_close(set(round(c.bbox[0]) for c in t.columns) |
                         set(round(c.bbox[2]) for c in t.columns))
    return bounds, t.bbox


def col_index(bounds, x):
    for i in range(len(bounds) - 1):
        if bounds[i] <= x < bounds[i + 1]:
            return i
    return 0 if x < bounds[0] else len(bounds) - 2


def clean(txt):
    txt = txt.replace("\ufffd", "'")
    return re.sub(r"[ \t]+", " ", txt).strip()


def build_names(n):
    if n == 10:
        return ["no", "tanggal", "time", "lat", "long", "depth", "mag",
                "wilayah", "korban", "sumber"]
    if n == 9:
        return ["no", "tanggal", "time", "koordinat", "depth", "mag",
                "wilayah", "korban", "sumber"]
    return [f"c{i}" for i in range(n)]


def extract():
    pdf = pdfplumber.open(PATH)
    pages = {p: pdf.pages[p - 1] for p in range(FIRST_PAGE, LOOKAHEAD + 1)}

    meta = {}
    for p, page in pages.items():
        bounds, bbox = main_table_columns(page)
        if bounds is None:
            meta[p] = None
            continue
        meta[p] = (bounds, build_names(len(bounds) - 1), bbox)

    # ---- build word stream assigned to columns ----
    raw = []          # (page, top, x, ci, text)
    for p, page in pages.items():
        m = meta[p]
        if m is None:
            continue
        bounds, names, bbox = m
        words = page.extract_words()
        # lines containing the "TAHUN <year>" heading -> drop
        lines = defaultdict(list)
        for w in words:
            lines[round(w["top"] / 3.0)].append(w)
        drop = set()
        for ws in lines.values():
            if any(w["text"] == "TAHUN" for w in ws):
                for w in ws:
                    drop.add((round(w["top"], 1), round(w["x0"], 1)))
        for w in words:
            if w["top"] > FOOTER_TOP:
                continue
            if (round(w["top"], 1), round(w["x0"], 1)) in drop:
                continue
            ci = col_index(bounds, (w["x0"] + w["x1"]) / 2)
            raw.append((p, round(w["top"], 1), round(w["x0"], 1), ci, w["text"]))
    raw.sort(key=lambda t: (t[0], t[1], t[2]))

    # ---- anchors: record's first line = earliest of its date line or time line ----
    groups = defaultdict(list)
    for it in raw:
        p, top, x, ci, text = it
        names = meta[p][1]
        if names[ci] in ("tanggal", "time"):
            groups[(p, names[ci], round(top / 3.0))].append(it)

    date_tops = []
    time_tops = []
    for (p, nm, _), ws in groups.items():
        ws.sort(key=lambda t: t[2])
        line = " ".join(t[4] for t in ws)
        top = min(t[1] for t in ws)
        if nm == "tanggal" and (DATE_LINE.match(line) or DATE_ABBR.match(line)):
            date_tops.append((p, top))
        if nm == "time" and re.match(r"^\d{1,2}[:.]\d{2}", line):
            time_tops.append((p, top))

    # a record is anchored by its date line; if its time sits on the line just
    # above the date (e.g. "14:55:0" / "2 September 2009"), start at the time line
    anchors_all = set()
    for (p, t2) in date_tops:
        t = t2
        for (p2, tt) in time_tops:
            if p2 == p and t2 - 22 <= tt < t2:
                t = min(t, tt)
        anchors_all.add((p, t))
    anchors_all = sorted(anchors_all)

    first_anchor = {}
    for p, top in anchors_all:
        if p not in first_anchor:
            first_anchor[p] = top

    # ---- drop the header band above the first record on each page ----
    stream = []
    for it in raw:
        p, top, x, ci, text = it
        bbox_top = meta[p][2][1]
        if p in first_anchor:
            if bbox_top - 2 <= top < first_anchor[p] - 6:
                continue
        else:
            if bbox_top - 2 <= top < bbox_top + 110 and text in HEADER_LABELS:
                continue
        stream.append(it)
    stream.sort(key=lambda t: (t[0], t[1], t[2]))
    keys = [(s[0], s[1]) for s in stream]

    anchors = [(p, top) for (p, top) in anchors_all if FIRST_PAGE <= p <= LAST_PAGE]

    # ---- assemble records ----
    recs = []
    for i, (p, top) in enumerate(anchors):
        start_j = bisect.bisect_left(keys, (p, top - TOL))
        if i + 1 < len(anchors):
            p2, t2 = anchors[i + 1]
            end_j = bisect.bisect_left(keys, (p2, t2 - TOL))
        else:
            end_j = len(stream)
        cols = defaultdict(list)
        for j in range(start_j, end_j):
            ip, itop, ix, ic, itext = stream[j]
            cols[meta[ip][1][ic]].append((ip, itop, ix, itext))
        recs.append({"page": p, "cols": cols})
    pdf.close()
    return recs


def col_text(items):
    items = sorted(items, key=lambda t: (t[0], t[1], t[2]))
    out, cur, line = [], None, []
    for ip, itop, ix, text in items:
        if cur is None or abs(itop - cur) > 3:
            if line:
                out.append(" ".join(line))
            line, cur = [text], itop
        else:
            line.append(text)
    if line:
        out.append(" ".join(line))
    return "\n".join(out)


def parse_coords(text):
    """Extract coordinate values, converting LS/BB hemispheres to a negative sign."""
    t = re.sub(r"\s+", "", text).upper()
    out = []
    for num, hemi in re.findall(r"(-?\s*\d+(?:\.\d+)?)\s*(LS|LU|BT|BB)?", t):
        s = num.replace(" ", "")
        if not s or s in ("-", "."):
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        if hemi in ("LS", "BB"):
            v = -abs(v)
        elif hemi in ("LU", "BT"):
            v = abs(v)
        out.append("%g" % v)
    return out


def parse_date(raw):
    """Return (tanggal_str, wilayah_str, year) from the Tanggal/Wilayah text."""
    flat = raw.replace("\n", " ")
    m = re.match(rf"^(\d{{1,2}})\s+({MONTHS})", flat)
    if m:
        ym = re.search(r"\d{4}", flat[m.start():m.start() + 45])
        if ym:
            end = m.start() + ym.end()
            tanggal = flat[:end].strip()
            wilayah = flat[end:].strip()
        else:
            tanggal = m.group(0)
            wilayah = flat[m.end():].strip()
        y = re.search(r"\d{4}", tanggal)
        return clean(tanggal), clean(wilayah), (y.group(0) if y else "")
    first_line = raw.split("\n", 1)[0].strip()
    ma = DATE_ABBR.match(first_line)
    if ma:
        mon = ABBR_MONTH.get(ma.group(2).capitalize(), ma.group(2))
        year = "20" + ma.group(3)
        wilayah = raw.split("\n", 1)[1] if "\n" in raw else ""
        return f"{ma.group(1)} {mon} {year}", clean(wilayah), year
    return "", clean(raw), ""


def main():
    recs = extract()

    # merge records whose "No" cell is empty into the previous record
    merged = []
    for r in recs:
        no_text = clean(col_text(r["cols"].get("no", [])))
        if not no_text and merged:
            for k, items in r["cols"].items():
                merged[-1]["cols"][k].extend(items)
            continue
        merged.append(r)

    rows = []
    for r in merged:
        cols = r["cols"]

        def get(nm):
            return clean(col_text(cols.get(nm, [])))

        def flat(nm):
            return clean(col_text(cols.get(nm, [])).replace("\n", " "))

        no_text = get("no")
        mno = re.match(r"(\d+)", no_text)
        no = int(mno.group(1)) if mno else ""
        tanggal, wilayah, tahun = parse_date(col_text(cols.get("tanggal", [])))
        waktu = flat("time")
        waktu = re.sub(r"(\d{1,2}[:.]\d{2}[:.]\d)\s+(\d)\b", r"\1\2", waktu)
        if "koordinat" in cols:
            coords = parse_coords(flat("koordinat"))
            lat = coords[0] if len(coords) > 0 else ""
            lon = coords[1] if len(coords) > 1 else ""
        else:
            clat = parse_coords(flat("lat"))
            clon = parse_coords(flat("long"))
            lat = clat[0] if clat else ""
            lon = clon[0] if clon else ""
        def num(nm):
            m2 = re.search(r"\d+(?:\.\d+)?", flat(nm))
            return m2.group(0) if m2 else ""

        rows.append({
            "Tahun": tahun, "No": no, "Tanggal": tanggal, "Wilayah": wilayah,
            "Origin_Time": waktu, "Latitude": lat, "Longitude": lon,
            "Depth_km": num("depth"), "Magnitude": num("mag"),
            "Wilayah_yang_merasakan": get("wilayah"),
            "Korban_Kerusakan": get("korban"), "Sumber": flat("sumber"),
        })
    rows.sort(key=lambda x: (x["Tahun"] or "0", x["No"] if isinstance(x["No"], int) else 0))

    fields = ["Tahun", "No", "Tanggal", "Wilayah", "Origin_Time", "Latitude",
              "Longitude", "Depth_km", "Magnitude", "Wilayah_yang_merasakan",
              "Korban_Kerusakan", "Sumber"]
    with open("gempabumi_2000-2023.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("rows:", len(rows))


if __name__ == "__main__":
    main()
