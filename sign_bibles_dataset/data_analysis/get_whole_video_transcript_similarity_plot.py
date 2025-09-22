import altair as alt
import pandas as pd
from pathlib import Path

import pandas as pd
import altair as alt

import numpy as np
import pandas as pd
import altair as alt


def get_global_transcript_similarity_heatmap(
    df,
    df_long,
    width=2000,
    height=2000,
):
    # Create Altair heatmap
    heatmap = (
        alt.Chart(df_long)
        .mark_rect()
        .encode(
            x=alt.X(
                "Reference:N",
                sort=None,
                title="Reference Segment",
                axis=alt.Axis(labelAngle=30),  # or 60, adjust as needed
            ),
            y=alt.Y("Query:N", sort=None, title="Query Segment"),
            color=alt.Color("Similarity:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["Query", "Reference", "Similarity"],
        )
        .properties(
            title="Transcript Segment Similarity (Global)",
            # width=width,
            # height=height
        )
    )
    if height is not None:
        heatmap = heatmap.properties(height=height)
    if width is not None:
        heatmap = heatmap.properties(width=width)

    # Add similarity numbers as text
    text = (
        alt.Chart(df_long)
        .mark_text(baseline="middle", fontSize=9)
        .encode(
            x="Reference:N",
            y="Query:N",
            text=alt.Text("Similarity:Q", format=".2f"),
            color=alt.condition("datum.Similarity > 0.5", alt.value("white"), alt.value("black")),
        )
    )

    return heatmap + text



def get_global_transcript_similarity_heatmap_sorted(
    df: pd.DataFrame,
    df_long: pd.DataFrame,
    focus_segment: str,
    width: int = 2000,
    height: int = 2000,
) -> alt.Chart:
    """
    Plot a similarity heatmap sorted by similarity to a given reference segment.

    Args:
        df: Wide similarity DataFrame (Query as first column).
        df_long: Long-format DataFrame with columns ['Query', 'Reference', 'Similarity'].
        focus_segment: Label of the reference segment to sort by.
        width: Chart width in pixels.
        height: Chart height in pixels.

    Returns:
        Altair heatmap chart.
    """

    # --- Sanity check ---
    if focus_segment not in df_long["Reference"].values:
        raise ValueError(f"Focus segment '{focus_segment}' not found in Reference column.")

    # --- Build sort order by filtering df_long for focus_segment as reference ---
    similarity_to_focus = (
        df_long[df_long["Reference"] == focus_segment]
        .sort_values("Similarity", ascending=False)
        .reset_index(drop=True)
    )
    sort_order = similarity_to_focus["Query"].tolist()

    # Create heatmap
    heatmap = (
        alt.Chart(df_long)
        .mark_rect()
        .encode(
            x=alt.X(
                "Reference:N",
                sort=sort_order,
                title=f"Reference Segment (sorted by similarity to '{focus_segment}')",
                axis=alt.Axis(labelAngle=30),
            ),
            y=alt.Y(
                "Query:N",
                sort=sort_order,
                title="Query Segment",
            ),
            color=alt.Color("Similarity:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["Query", "Reference", "Similarity"],
        )
        .properties(
            title=f"Transcript Similarity Sorted by '{focus_segment}'",
            width=width,
            height=height,
        )
    )

    # Add text overlay with similarity values
    text = (
        alt.Chart(df_long)
        .mark_text(baseline="middle", fontSize=9)
        .encode(
            x=alt.X("Reference:N", sort=sort_order),
            y=alt.Y("Query:N", sort=sort_order),
            text=alt.Text("Similarity:Q", format=".2f"),
            color=alt.condition(
                "datum.Similarity > 0.5", alt.value("white"), alt.value("black")
            ),
        )
    )

    return heatmap + text





if __name__ == "__main__":
    df = pd.read_parquet(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/data_analysis/text_similarity/whole_videos_similarity_matrix.parquet"
    )
    df_long = pd.read_parquet(
        "/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/data_analysis/text_similarity/whole_videos_similarity_matrix_long.parquet"
    )


    heatmap = get_global_transcript_similarity_heatmap(df, df_long)


    focus_segment = "1: God Creates The World"
    heatmap_sorted = get_global_transcript_similarity_heatmap_sorted(
        df=df,
        df_long=df_long,
        focus_segment=focus_segment,
        width=2500,
        height=2500,
    )

    out_dir = Path("/opt/home/cleong/projects/semantic_and_visual_similarity/sign-bibles-dataset/sign_bibles_dataset/data_analysis/text_similarity")
    heatmap.save(out_dir/f"whole_videos_similarity_all-MiniLM-L6-v2.pdf")
    heatmap.save(out_dir/f"whole_videos_similarity_all-MiniLM-L6-v2.png")
    heatmap.save(out_dir/f"whole_videos_similarity_all-MiniLM-L6-v2.json")



    heatmap_sorted.save(out_dir/f"whole_videos_similarity_all-MiniLM-L6-v2_sorted.pdf")
    heatmap_sorted.save(out_dir/f"whole_videos_similarity_all-MiniLM-L6-v2_sorted.png")
    heatmap_sorted.save(out_dir/f"whole_videos_similarity_all-MiniLM-L6-v2_sorted.json")
    
