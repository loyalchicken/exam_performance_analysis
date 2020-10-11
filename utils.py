import numpy as np

def add_median_label_on_boxplot(data, plot, x, y, x_ticks):
    medians = data.groupby([x])[y].median().round(2)
    vertical_offset = data[y].median() * 0.02 # offset from median for display
    for i in range(len(x_ticks)):
        if len(data[data[x]==x_ticks[i]])>0:
            plot.text(i, medians[x_ticks[i]] + vertical_offset, medians[x_ticks[i]], horizontalalignment='center',size=10,color='black',weight='light')

def standardize(data, num_entries_threshold):
    d = data.copy()
    d = d[d["Number of Submissions"]>=num_entries_threshold][["Median Time Before Deadline", "View Per Exam", "View Per HW", "HW %", "Number of Submissions", "Gender"]]
    d = (d-np.mean(d,axis=0)) / np.std(d, axis=0)
    return d

def normalize(data, num_entries_threshold):
    d = data.copy()
    d = d[d["Number of Submissions"]>=num_entries_threshold][["Median Time Before Deadline", "View Per Exam", "View Per HW", "HW %", "Number of Submissions", "Scaled Exam Score"]]
    return d / np.linalg.norm(d)

