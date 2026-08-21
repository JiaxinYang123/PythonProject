from pathlib import Path
from typing import Iterator

import ijson


def iterate_weekly_snapshots(
    json_path: Path,
) -> Iterator[tuple[str, list, list]]:
    """
    Stream weekly snapshots from the official DeXposure JSON file.

    The network structure and source-target direction are taken directly
    from the processed DeXposure dataset. This function does not rebuild
    or modify the underlying protocol network.

    Parameters
    ----------
    json_path:
        Path to a DeXposure historical-network JSON file.

    Yields
    ------
    date:
        Weekly snapshot date in YYYY-MM-DD format.
    nodes:
        Nodes supplied in the official weekly snapshot.
    links:
        Directed weighted links supplied in the official weekly snapshot.
    """

    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"DeXposure file not found: {json_path}"
        )

    with json_path.open("rb") as file:

        for date, snapshot in ijson.kvitems(
            file,
            "data",
            use_float=True,
        ):
            if "nodes" not in snapshot:
                raise KeyError(
                    f"Snapshot {date} does not contain 'nodes'."
                )

            if "links" not in snapshot:
                raise KeyError(
                    f"Snapshot {date} does not contain 'links'."
                )

            yield (
                date,
                snapshot["nodes"],
                snapshot["links"],
            )