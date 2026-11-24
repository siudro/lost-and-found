"""Check which dataset sources are reachable right now.

The brief says to verify links rather than trust them. This script does that
mechanically so the research document cannot drift out of date without it
being noticed.

Run: python tools/check_dataset_access.py

It reports HTTP status only. It downloads nothing and needs no credentials.
"""

from __future__ import annotations

import urllib.error
import urllib.request

SOURCES = {
    "CAVIAR index (best find, CC BY-SA)": "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/",
    "CAVIAR LeftBag_AtChair (verified)": "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/LeftBag_AtChair/LeftBag_AtChair.mpg",
    "CAVIAR LeftBag_PickedUp (verified)": "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/LeftBag_PickedUp/LeftBag_PickedUp.mpg",
    "UniData HF mirror (usable)": "https://huggingface.co/api/datasets/UniDataPro/abandoned-object-detection-video",
    "UniData first clip (usable)": "https://huggingface.co/datasets/UniDataPro/abandoned-object-detection-video/resolve/main/left-behined_item_0001.mp4",
    "Kaggle listing (paid preview)": "https://www.kaggle.com/datasets/unidpro/abandoned-object-detection-video-dataset",
    "ABODA original (dead/squatted)": "http://www.mikelrodriguez.com/abandoned-object-database/",
    "PETS 2006 (unreachable)": "https://www.cvg.reading.ac.uk/PETS2006/",
    "AVSS 2007 (unreachable)": "https://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html",
    "UCF-Crime (negatives)": "https://www.crcv.ucf.edu/projects/real-world/",
    "DeepFashion2 masks (appearance)": "https://huggingface.co/api/datasets/zyuzuguldu/deepfashion2-upper-body-masks",
}

TIMEOUT_SECONDS = 15


def probe(url: str) -> str:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "lost-and-found-dataset-check"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return f"HTTP {response.status}"
    except urllib.error.HTTPError as error:
        return f"HTTP {error.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return f"UNREACHABLE ({type(error).__name__})"


def main() -> None:
    width = max(len(name) for name in SOURCES)
    for name, url in SOURCES.items():
        print(f"{name.ljust(width)}  {probe(url)}")
    print("\nHTTP 200 on a landing page does not mean the dataset is there.")
    print("ABODA returns 200 but serves a domain-squatter redirect; that is")
    print("why it is marked dead in docs/datasets.md.")


if __name__ == "__main__":
    main()
