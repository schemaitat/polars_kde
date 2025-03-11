/// This module provides functions for performing kernel density estimation (KDE) on Polars Series.
///
/// # Functions
///
/// - `same_output_type`: A helper function that returns the same output type as the input fields.
/// - `kde_dynamic_evals`: Applies KDE to a series of sample points and evaluation points, returning the resulting density estimates as a series.
/// - `kde_static_evals`: Applies KDE to a series of sample points with evaluation points provided via keyword arguments, returning the resulting density estimates as a series.
/// - `kde_agg`: Aggregates KDE results for a series of sample points with evaluation points provided via keyword arguments, returning the resulting density estimates as a series.
///
/// # Structs
///
/// - `KdeKwargs`: A struct for holding keyword arguments for KDE functions, specifically the evaluation points.
///
/// # Example
///
/// ```rust
/// use kernel_density_estimation::prelude::*;
/// use polars::prelude::*;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct KdeKwargs {
///     eval_points: Vec<f32>,
/// }
///
/// fn main() -> PolarsResult<()> {
///     // Example usage of kde function
///     let sample_series = Series::new("samples", vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
///     let kwargs = KdeKwargs { eval_points: vec![1.0, 2.0, 3.0] };
///     let result = kde(&[sample_series], kwargs)?;
///     println!("{:?}", result);
///     Ok(())
/// }
/// ```
use kernel_density_estimation::prelude::*;
use polars::prelude::*;
use polars_core::utils::align_chunks_binary;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// A struct for holding keyword arguments for KDE functions, specifically the evaluation points.
#[derive(Deserialize)]
struct KdeKwargs {
    eval_points: Vec<f32>,
}

/// A helper function that returns the same output type as the input fields.
///
/// # Arguments
///
/// * `input_fields` - A slice of input fields.
///
/// # Returns
///
/// A result containing the same output type as the input fields.
fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

/// Computes the kernel density estimation (KDE) for given sample points and evaluation points.
///
/// # Arguments
///
/// * `sample_points` - A vector of sample points.
/// * `eval_points` - A vector of evaluation points.
///
/// # Returns
///
/// A vector containing the KDE density estimates.
fn compute_kde(sample_points: Vec<f32>, eval_points: Vec<f32>) -> Vec<f32> {
    if sample_points.len() <= 1 {
        return vec![0.0; eval_points.len()];
    }

    let kde = KernelDensityEstimator::new(sample_points, Silverman, Normal);
    kde.pdf(&eval_points)
}

/// Applies KDE to a series of sample points and evaluation points, returning the resulting density estimates as a series.
///
/// # Arguments
///
/// * `inputs` - A slice of input series.
///
/// # Returns
///
/// A result containing the series with the KDE density estimates.
#[polars_expr(output_type_func=same_output_type)]
fn kde_dynamic_evals(inputs: &[Series]) -> PolarsResult<Series> {
    let sample_points: &ListChunked = inputs[0].list()?;
    let eval_points: &ListChunked = inputs[1].list()?;

    polars_ensure!(
        sample_points.dtype() == &DataType::List(Box::new(DataType::Float32)),
        ComputeError: "Expected `values` to be of type `List(Float32)`, got: {}", sample_points.dtype()
    );

    let (sample_points, eval_points) = align_chunks_binary(sample_points, eval_points);

    let out: ListChunked = sample_points
        .amortized_iter()
        .zip(eval_points.amortized_iter())
        .map(|(lhs, rhs)| {
            let lhs = lhs.unwrap();
            let rhs = rhs.unwrap();

            let points_inner: &Float32Chunked = lhs.as_ref().f32().unwrap();
            let eval_innter: &Float32Chunked = rhs.as_ref().f32().unwrap();

            let sample_points = points_inner.into_no_null_iter().collect::<Vec<_>>();

            let eval_points = eval_innter.into_no_null_iter().collect::<Vec<_>>();

            let samples = compute_kde(sample_points, eval_points);

            Series::new(PlSmallStr::EMPTY, samples)
        })
        .collect();

    Ok(out.into_series())
}

/// Applies KDE to a series of sample points with evaluation points provided via keyword arguments, returning the resulting density estimates as a series.
///
/// # Arguments
///
/// * `inputs` - A slice of input series.
/// * `kwargs` - A struct containing the evaluation points.
///
/// # Returns
///
/// A result containing the series with the KDE density estimates.
#[polars_expr(output_type_func=same_output_type)]
fn kde_static_evals(inputs: &[Series], kwargs: KdeKwargs) -> PolarsResult<Series> {
    let ca: &ListChunked = inputs[0].list()?;

    polars_ensure!(
        ca.dtype() == &DataType::List(Box::new(DataType::Float32)),
        ComputeError: "Expected `values` to be of type `List(Float32)`, got: {}", ca.dtype()
    );

    let eval_points = kwargs.eval_points;

    let out: ListChunked = ca.apply_amortized(|s| {
        let s = s.as_ref();
        let points_inner = s.f32().unwrap();

        let sample_points = points_inner.into_no_null_iter().collect::<Vec<_>>();

        let samples = compute_kde(sample_points, eval_points.clone());

        Series::new(PlSmallStr::EMPTY, samples)
    });

    Ok(out.into_series())
}

/// Aggregates KDE results for a series of sample points with evaluation points provided via keyword arguments, returning the resulting density estimates as a series.
///
/// # Arguments
///
/// * `inputs` - A slice of input series.
/// * `kwargs` - A struct containing the evaluation points.
///
/// # Returns
///
/// A result containing the series with the KDE density estimates.
#[polars_expr(output_type_func=same_output_type)]
fn kde_agg(inputs: &[Series], kwargs: KdeKwargs) -> PolarsResult<Series> {
    let values = &inputs[0].f32()?;
    let eval_points = kwargs.eval_points;

    let sample_points = values.into_no_null_iter().collect::<Vec<_>>();

    let samples = compute_kde(sample_points, eval_points);

    Ok(Series::new(PlSmallStr::EMPTY, samples))
}
