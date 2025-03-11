# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "kaleido==0.2.1",
#     "marimo",
#     "nbformat==5.10.4",
#     "numpy==2.2.3",
#     "plotly==6.0.0",
#     "polars==1.24.0",
#     "scipy==1.15.2",
# ]
# ///

import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    from functools import partial, wraps
    from itertools import product
    import polars as pl
    from scipy.stats import gaussian_kde
    import polars_kde as pkde
    import numpy as np
    import time
    import kaleido

    def benchmark(f, *args, **kwargs):
        ts = time.time()
        _ = f(*args, **kwargs)
        te = time.time()

        name = f.__name__
        total_time = te - ts

        return pl.DataFrame(
            {
                "name": [name],
                "total_time": [total_time],
                "n_rows": [kwargs.get("df").height],
                "n_groups": [kwargs.get("df").n_unique("group")],
                "n_eval_points": [len(kwargs.get("eval_points"))],
            }
        )

    def get_df(
        n_points: int = 1000,
        n_groups: int = 10,
    ) -> pl.LazyFrame:
        return (
            pl.LazyFrame(
                {"a": np.random.normal(0, 1, size=n_points)},
                schema=pl.Schema({"a": pl.Float32}),
            )
            .with_columns(
                group=pl.cum_count("a").qcut(n_groups),
            )
            .lazy()
        )

    def kde_static_evals(df: pl.DataFrame, eval_points: list[float]) -> pl.DataFrame:
        return (
            df.group_by("group")
            .agg(pl.col("a").cast(pl.Float32))
            .with_columns(
                kde=pkde.kde_static_evals(pl.col("a"), eval_points=eval_points)
            )
        )

    def kde_scipy(df: pl.DataFrame, eval_points: list[float]) -> pl.DataFrame:
        def get_kde(arr: np.array) -> np.array:
            try:
                g_kde = gaussian_kde(arr)
                kde_points = g_kde.evaluate(eval_points)
                return kde_points
            except Exception as e:
                print(e)
                return np.array([0])

        return (
            df.group_by("group")
            .agg(
                pl.col("a").cast(pl.Float32),
            )
            .with_columns(
                kde=pl.col("a").map_elements(
                    get_kde,
                    return_dtype=pl.List(pl.Float32),
                )
            )
        )

    def kde(df: pl.DataFrame, eval_points: list[float]) -> pl.DataFrame:
        return df.group_by("group").agg(
            kde=pkde.kde(pl.col("a"), eval_points=eval_points)
        )

    return (
        benchmark,
        gaussian_kde,
        get_df,
        kaleido,
        kde,
        kde_scipy,
        kde_static_evals,
        mo,
        np,
        partial,
        pkde,
        pl,
        product,
        px,
        time,
        wraps,
    )


@app.cell
def _(benchmark, get_df, kde, kde_scipy, kde_static_evals, np, pl, product):
    eval_points = np.linspace(0, 1, 50).tolist()

    # df_kde = kde(get_df(10_000_000), eval_points=eval_points)

    functions = [kde_scipy, kde, kde_static_evals]
    test_frames = [
        get_df(i, j)
        for i in [10_000, 100_000, 500_000, 1_000_000, 5_000_000]
        for j in [1, 10, 100, 1000]
    ]

    results: list[pl.LazyFrame] = []

    for f, df in product(functions, test_frames):
        df_c = df.collect()
        print("Starting benchmark for", f.__name__, "with", df_c.height, "rows.")
        results.append(benchmark(f, df=df_c, eval_points=eval_points))

    df_bench = pl.concat(results)
    return df, df_bench, df_c, eval_points, f, functions, results, test_frames


@app.cell
def _(df_bench, px):
    fig = px.bar(
        df_bench.sort("n_groups"),
        x="n_rows",
        y="total_time",
        color="name",
        facet_row="n_groups",
        barmode="group",
    )
    fig.update_layout(
        title="KDE Benchmark",
        xaxis_title="Number of Rows",
        yaxis_title="Total Time (s)",
        height=1000,
    )
    fig.update_xaxes(type="category")
    fig
    return (fig,)


@app.cell
def _(fig):
    with open("benchmark.png", "wb") as file:
        file.write(fig.to_image(format="png"))
    return (file,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
