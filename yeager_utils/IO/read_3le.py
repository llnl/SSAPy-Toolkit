import re
import numpy as np
import pandas as pd

def read_3le(data_file, makedf=True, verbose=False):
    """
    Parse a 3-line element (0/1/2) text file into a dict of lists or a DataFrame.

    Parameters
    ----------
    data_file : str | Path
        Input file path.
    makedf : bool, default True
        If True, return a pandas DataFrame with one row per record (aligned by line '1').
        If False, return the raw dict of lists accumulated while parsing.

    Returns
    -------
    pandas.DataFrame | dict
    """
    def split_line(line):
        payload = re.findall(r'"[^"]+"|\'[^\']+\'|\S+', line)
        return [s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'" else s for s in payload]

    def _days_in_year(y: int) -> float:
        # Leap year if divisible by 4, except centuries not divisible by 400
        return 366.0 if ((y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)) else 365.0

    def _century_year(two_digit: str) -> int:
        """Convert 2-digit year to 4-digit using Space-Track pivot (>=57 -> 1900s, else 2000s)."""
        yy = int(two_digit)
        return 1900 + yy if yy >= 57 else 2000 + yy

    def vprint(message):
        if verbose:
            print(message)

    with open(data_file, "r", encoding="utf-8", errors="replace") as f:
        def split_mantassa(text):
            t = text.strip()
            if not t or t.strip('0+-') == '':
                return 0.0
            # sanity check: last two chars should be like '+5' or '-2'
            if len(t) < 3 or t[-2] not in '+-' or not t[-1].isdigit():
                raise ValueError(f"Bad TLE exp field: {text!r}")
            sign = '-' if t[0] == '-' else ''
            mant = t[1:-2] if t[0] in '+-' else t[:-2]
            return float(f"{sign}0.{mant}e{t[-2:]}")

        # detect if a token is an International Designator (YYNNNPPP)
        def looks_like_intl(tok: str) -> bool:
            return bool(re.fullmatch(r"\d{2}\d{3}[A-Za-z]{1,3}", tok))

        # parse epoch into epoch_year and epoch_day (Space-Track rules)
        def parse_epoch_fields(epoch_token: str):
            t = epoch_token.replace(" ", "")
            yy = int(t[:2])
            epoch_year = (1900 + yy) if yy >= 57 else (2000 + yy)
            epoch_day = float(t[2:])  # DDD.dddddd
            return epoch_year, epoch_day

        cols = [
            "name",
            "satellite#",
            "classification",
            "intl_designator",
            "launch_year",
            "launch_number",
            "launch_piece",
            "elset_epoch",
            "epoch_year",
            "epoch_day",
            "decimal_year",
            "ndot",
            "nddot",
            "drag",
            "elset_type",
            "elset#",
            "catalog number",
            "inc_deg",
            "raan_deg",
            "ecc",
            "pa_deg",
            "mean_anomaly_deg",
            "mean_motion_deg",
            "mean_motion_rev_day_deg",
            "epoch_rev",
            "check_sum",
        ]

        data = {}
        for key in cols:
            vprint(f"Adding {key} to data.")
            data[key] = []

        for raw in f:
            # Strip newline characters
            line = raw.rstrip("\r\n")

            # Handle potential UTF-8 BOM on the very first line
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")

            if line == "":
                continue  # skip blank lines

            key = line[0]
            rest = line[1:]

            splits = split_line(rest)
            vprint(splits)
            if key == "0":
                data["name"].append(splits)

            elif key == "1":
                if not splits:
                    continue
                # first token like "ISSU" -> satellite name and classification
                data["satellite#"].append(splits[0][:-1])
                data["classification"].append(splits[0][-1])

                # detect optional International Designator and shift indices if needed
                has_intl = len(splits) > 1 and looks_like_intl(splits[1])
                i = 0 if has_intl else -1

                if has_intl:
                    data["intl_designator"].append(splits[1 + i])
                    # Convert 2-digit launch year to 4-digit
                    data["launch_year"].append(int(splits[1 + i][:2]))
                    data["launch_number"].append(int(splits[1 + i][2:5]))
                    data["launch_piece"].append(splits[1 + i][5:])
                else:
                    data["intl_designator"].append(None)
                    data["launch_year"].append(None)
                    data["launch_number"].append(None)
                    data["launch_piece"].append(None)

                # epoch fields
                data["elset_epoch"].append(splits[2 + i])
                ey, ed = parse_epoch_fields(splits[2 + i])
                data["epoch_year"].append(ey)
                data["epoch_day"].append(ed)

                # decimal year (use numpy-friendly floats)
                denom = _days_in_year(int(ey))
                dec_year = float(ey) + (float(ed) - 1.0) / float(denom)
                data["decimal_year"].append(dec_year)

                data["ndot"].append(float(splits[3 + i]))
                data["nddot"].append(split_mantassa(splits[4 + i]))
                data["drag"].append(split_mantassa(splits[5 + i]))
                data["elset_type"].append(int(splits[6 + i]))
                data["elset#"].append(int(splits[7 + i]))

            elif key == "2":
                if len(splits) < 6:
                    continue
                data["inc_deg"].append(float(splits[1]))
                data["raan_deg"].append(float(splits[2]))
                data["ecc"].append(float(splits[3]))
                data["pa_deg"].append(float(splits[4]))
                data["mean_anomaly_deg"].append(float(splits[5]))
                if len(splits) >= 7 and len(splits[6]) > 11:
                    s6 = splits[6]
                    data["mean_motion_deg"].append(float(s6[:11]))
                    data["epoch_rev"].append(int(s6[11:15]))
                    data["check_sum"].append(int(s6[15]))
            else:
                # Unrecognized record type; ignore or log as needed
                pass

    if not makedf:
        return data

    # ---------- Build a DataFrame (align primarily on the count/order of line '1') ----------
    base_n = len(data[cols[0]])  # line '1' count drives row count

    # helper to safely fetch i-th element with NaN fallback
    def _get(k, i, default=np.nan):
        lst = data.get(k, [])
        return lst[i] if i < len(lst) else default

    rows = []
    for i in range(base_n):
        row = {k: _get(k, i) for k in cols}
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    return df
