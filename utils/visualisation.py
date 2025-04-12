
# List of job titles of interest
job_titles_of_interest = ['Consultant', 'Senior Consultant', 'Manager', 'Associate Director', 'Director', 'Partner']

# Filtering the DataFrame for the job titles of interest
filtered_df = df[df['jobTitle'].isin(job_titles_of_interest)]

# Set up the figure and axes
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))
axes = axes.flatten()

# Loop through each job title of interest and plot the distribution
for i, title in enumerate(job_titles_of_interest):
    ax = axes[i]
    title_df = filtered_df[filtered_df['jobTitle'] == title]

    sns.histplot(title_df['jobTenureYears'], kde=True, bins=20, edgecolor='black', alpha=0.7, ax=ax)

    # Fit a normal distribution and plot it
    mu, std = stats.norm.fit(title_df['jobTenureYears'].dropna())
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)

    # Annotate the plot with the distribution type
    ax.set_title(f'{title}\n(mu={mu:.2f}, std={std:.2f})')
    ax.set_xlabel('Job Tenure Years')
    ax.set_ylabel('Frequency')

# Adjust the layout
plt.tight_layout()
plt.show()
