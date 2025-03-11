import pytest
import polars as pl
import polars_kde as pkde
from polars.testing import assert_series_equal


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "id": [0, 0, 1, 1, 1],
        },
        schema=pl.Schema(
            {
                "a": pl.Float32,
                "id": pl.Int64,
            }
        ),
    )


@pytest.fixture
def eval_points() -> list[float]:
    return [1.0, 2.0, 3.0, 4.0, 5.0]


def test_kde_static_evals(sample_df, eval_points):
    df = sample_df.group_by("id").agg(pl.col("a"))

    df_kde = df.with_columns(
        kde=pkde.kde_static_evals(
            pl.col("a"),
            eval_points=eval_points,
        )
    )

    assert df_kde.shape == (2, 3)
    assert df_kde.select("kde").dtypes[0] == pl.List(pl.Float32)

    assert_series_equal(
        df_kde["kde"].list.len(),
        pl.Series([5, 5]),
        check_names=False,
        check_dtypes=False,
    )


def test_kde_dynamic(
    sample_df,
):
    e1 = [1.0, 2.0, 3.0]
    e2 = [4.0, 5.0]

    df = (
        sample_df.group_by("id")
        .agg(pl.col("a"))
        .with_columns(eval_points=pl.Series([e1, e2]).cast(pl.List(pl.Float32)))
    )

    df_kde = df.with_columns(
        kde=pkde.kde_dynamic_evals(
            pl.col("a"),
            pl.col("eval_points"),
        )
    )

    assert df_kde.shape == (2, 4)
    assert df_kde.select("kde").dtypes[0] == pl.List(pl.Float32)

    # the group with id=0 has 3 eval points
    # the group with id=1 has 2 eval points
    assert_series_equal(
        df_kde["kde"].list.len(),
        pl.Series([3, 2]),
        check_names=False,
        check_dtypes=False,
    )


def test_kde_agg(
    sample_df,
    eval_points,
):
    df_kde = sample_df.group_by("id").agg(
        kde=pkde.kde(
            pl.col("a"),
            eval_points=eval_points,
        )
    )

    assert df_kde.shape == (2, 2)
    assert df_kde.select("kde").dtypes[0] == pl.List(pl.Float32)

    assert_series_equal(
        df_kde["kde"].list.len(),
        pl.Series([5, 5]),
        check_names=False,
        check_dtypes=False,
    )
