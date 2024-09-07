# Databricks notebook source
# MAGIC %md
# MAGIC ### Plotting metrics

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC import pandas as pd

# COMMAND ----------

model = "dt" #change to lr if Linear regession
model_name = "Linear Regression" if model == "lr" else "Decision Tree Regression"

# COMMAND ----------

num_cores = [1, 2, 4, 8, 12] # for size-up and speed-up
sizes = [10, 20, 40, 80, 100]

# COMMAND ----------

def plot_scalability_measure(name, stats, xlabel, x_vals, linear_vals):
    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(
        f"{name.capitalize()} of a Local-based model with {model_name}"
    )

    # Plot the measure with and without overhead
    ax[0].plot(x_vals, stats[name], "k", label=f"Real {name}")
    ax[1].plot(x_vals, stats[f"{name}_no_overhead"], "k",
               label=f"{name} without overhead")

    # Plot the linear values for the measure
    for i in [0, 1]:
        ax[i].plot(x_vals, linear_vals, "--", color="gray",
            label=f"Linear {name}", linewidth=1)
        ax[i].set(xlabel=xlabel, ylabel=name.capitalize())
        if name == "scale-up":
            ax[i].set_ylim([0, 1.1])
        ax[i].legend()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot Size-up

# COMMAND ----------

def size_up(results, runtime_col):
  return results[runtime_col] / results[runtime_col].iloc[0]

# COMMAND ----------

results = pd.read_csv(f"sizeup_{model}.csv", names=["cores", "pct", "runtime", "runtime_no_overhead"])
avg_results = results.groupby('pct').mean()
avg_results['overhead'] = avg_results['runtime'] - avg_results['runtime_no_overhead']
avg_results

# COMMAND ----------

avg_results['size-up'] = size_up(avg_results, 'runtime')
avg_results['size-up_no_overhead'] = size_up(avg_results, 'runtime_no_overhead')
avg_results[['runtime', 'runtime_no_overhead', 'size-up',
            'size-up_no_overhead']]

linear_sizeup = [1, 2, 4, 8, 10]
plot_scalability_measure('size-up', avg_results, "Size of Data", sizes, linear_sizeup)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot Speed-up

# COMMAND ----------

def speed_up(results, runtime_col):
    return results[runtime_col].iloc[0] / results[runtime_col]

# COMMAND ----------

results = pd.read_csv(f'speedup_{model}.csv', names=['cores', 'pct',
                      'runtime', 'runtime_no_overhead'])
avg_results = results.groupby('cores').mean()
avg_results['overhead'] = avg_results['runtime'] \
    - avg_results['runtime_no_overhead']
avg_results

# COMMAND ----------

avg_results['speed-up'] = speed_up(avg_results, 'runtime')
avg_results['speed-up_no_overhead'] = speed_up(avg_results, 'runtime_no_overhead')
avg_results[['runtime', 'runtime_no_overhead', 'speed-up',
            'speed-up_no_overhead']]
avg_results

# COMMAND ----------

linear_sizeup = [1, 2, 4, 8, 10]
plot_scalability_measure('speed-up', avg_results, "Number of cores", num_cores, num_cores)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot Scale-up

# COMMAND ----------

num_cores = [1, 2, 4, 8, 10] # for scale-up

# COMMAND ----------

def scale_up(results, runtime_col):
    return results[runtime_col].iloc[0] / results[runtime_col]

# COMMAND ----------

results = pd.read_csv(f"scaleup_{model}.csv",
    names=["cores", "pct", "runtime", "runtime_no_overhead"],
)
avg_results = results.groupby(["cores", "pct"]).mean()
avg_results["overhead"] = (
    avg_results["runtime"] - avg_results["runtime_no_overhead"]
)
avg_results

# COMMAND ----------

avg_results['scale-up'] = scale_up(avg_results, 'runtime')
avg_results['scale-up_no_overhead'] = scale_up(avg_results, 'runtime_no_overhead')
avg_results[['runtime', 'runtime_no_overhead', 'scale-up', 'scale-up_no_overhead']]

ideal_scaleup = [1] * 5
plot_scalability_measure('scale-up', avg_results, 'Number of cores',
                         num_cores, ideal_scaleup)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot Linear and Non-linear models in the same graphs
# MAGIC

# COMMAND ----------

def plot_scalability_measure_both(name, stats, xlabel, x_vals, linear_vals):
    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(
        f"Comparison of {name.capitalize()} of a Local-based model with Linear Regression and Decision Tree"
    )

    # Plot the measure with and without overhead
    ax[0].plot(x_vals, stats[name+"_lr"], label=f"Real {name} of LR")
    ax[0].plot(x_vals, stats[name+"_dt"], label=f"Real {name} of DT")
    ax[1].plot(x_vals, stats[f"{name}_no_overhead_lr"], label=f"{name} without overhead of LR")
    ax[1].plot(x_vals, stats[f"{name}_no_overhead_dt"], label=f"{name} without overhead of DT")

    # Plot the linear values for the measure
    for i in [0, 1]:
        ax[i].plot(x_vals, linear_vals, "--", color="gray",
            label=f"Linear {name}", linewidth=1)
        ax[i].set(xlabel=xlabel, ylabel=name.capitalize())
        if name == "scale-up":
            ax[i].set_ylim([0, 1.1])
        ax[i].legend()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scale-up

# COMMAND ----------

results_lr = pd.read_csv(f'scaleup_lr.csv', names=['cores', 'pct',
                      'runtime_lr', 'runtime_no_overhead_lr'])
results_dt = pd.read_csv(f'scaleup_dt.csv', names=['cores', 'pct',
                      'runtime_dt', 'runtime_no_overhead_dt'])
results = results_lr.merge(results_dt, on=['cores', 'pct'], suffixes=(False, False))
avg_results = results.groupby(['cores', 'pct']).mean()
avg_results['overhead_lr'] = avg_results['runtime_lr'] - avg_results['runtime_no_overhead_lr']
avg_results['overhead_dt'] = avg_results['runtime_dt'] - avg_results['runtime_no_overhead_dt']
avg_results

# COMMAND ----------

avg_results['scale-up_lr'] = scale_up(avg_results, 'runtime_lr')
avg_results['scale-up_no_overhead_lr'] = scale_up(avg_results, 'runtime_no_overhead_lr')
avg_results['scale-up_dt'] = scale_up(avg_results, 'runtime_dt')
avg_results['scale-up_no_overhead_dt'] = scale_up(avg_results, 'runtime_no_overhead_dt')
avg_results[['runtime_lr', 'runtime_no_overhead_lr', 'scale-up_lr',
            'scale-up_no_overhead_lr', 'runtime_dt', 'runtime_no_overhead_dt', 'scale-up_dt',
            'scale-up_no_overhead_dt',]]
avg_results

# COMMAND ----------

ideal_scaleup = [1] * 5
plot_scalability_measure_both('scale-up', avg_results, "Number of cores", num_cores, ideal_scaleup)