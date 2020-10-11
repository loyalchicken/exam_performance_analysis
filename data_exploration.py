from utils import add_median_label_on_boxplot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def time_vs_exam_performance_plot(data, num_groups, num_entries_threshold):
    d=data.sort_values(by='Median Time Before Deadline', ascending=True)
    d=d[d["Number of Submissions"]>=num_entries_threshold]
    d=d[["Scaled Exam Score", "Median Time Before Deadline"]]
    d=d.reset_index(drop=True)
    num_per_group = len(d)/num_groups
    d["Group by Time of Submission"]=np.floor([i/num_per_group for i in range(len(d))])
    group_maxes = d[["Group by Time of Submission", "Median Time Before Deadline"]].groupby(by="Group by Time of Submission").max()
    group_mins = d[["Group by Time of Submission", "Median Time Before Deadline"]].groupby(by="Group by Time of Submission").min()

    plt.figure(figsize=(12,6))
    p=sns.boxplot(d["Group by Time of Submission"], d["Scaled Exam Score"], whis=True, showfliers=False)
    p.set_title("HW Submission Time Before Deadline vs. Exam Performance", fontsize=18)
    p.set_xlabel("Submission Time (Grouped)",fontsize=14)
    p.set_ylabel("Scaled Exam Score",fontsize=14)
    sns.set_style("whitegrid")
    add_median_label_on_boxplot(d, p, "Group by Time of Submission", "Scaled Exam Score", np.arange(0,num_groups))
    return p, group_maxes, group_mins

def mt1_views_vs_mt2_score_plot(data):
    d = data[data["View Count M1"] <40]
    plt.figure(figsize=(15,5))
    p=sns.boxplot(x="View Count M1", y="M2", data=d, whis=True, showfliers=False)
    add_median_label_on_boxplot(d, p, "View Count M1", "M2", p.get_xticks())

def mt2_views_vs_final_score_plot(data):
    d = data[data["View Count M2"] <60]
    plt.figure(figsize=(15,5))
    p=sns.boxplot(x="View Count M2", y="Final", data=d, whis=True, showfliers=False)
    add_median_label_on_boxplot(d, p, "View Count M2", "Final", p.get_xticks())

def views_per_exam_vs_exam_score(data):
    d = data.copy()
    plt.figure(figsize=(7,4))
    d = d[["View Per Exam", "Scaled Exam Score"]]
    d["View Per Exam"] = np.floor(d["View Per Exam"]).astype(int)
    hist = sns.distplot(d["View Per Exam"], kde=False) 
    hist.set(ylabel='Number of Students')
    plt.figure(figsize=(15,7))
    p = sns.boxplot(x="View Per Exam", y="Scaled Exam Score", data=d, whis=True, fliersize=0)
    add_median_label_on_boxplot(d, p, "View Per Exam", "Scaled Exam Score", p.get_xticks())

def hw_views_vs_exam_score(data, num_entries_threshold):
    d = data.copy()
    d = d[d["Number of Submissions"]>=num_entries_threshold]
    d["View Per HW Floored"]=np.floor(d["View Per HW"])
    plt.figure(figsize=(10,5))
    hist = sns.distplot(d["View Per HW Floored"], kde=False)
    hist.set(ylabel='Number of Students')
    plt.figure(figsize=(10,5))
    p = sns.boxplot(x="View Per HW Floored", y="Scaled Exam Score", data=d)
    add_median_label_on_boxplot(d, p, "View Per HW Floored", "Scaled Exam Score", p.get_xticks())

def hw_scores_vs_exam_scores(data, num_groups, num_entries_threshold):
    d = data.copy()
    d = d[["HW %", "Scaled Exam Score", "Number of Submissions"]]
    d=d[d["Number of Submissions"]>=num_entries_threshold]
    num_per_group = len(d)/num_entries_threshold
    d=d.sort_values(by="HW %")
    d["HW Percentile by Group"]=np.floor([i/num_per_group for i in range(len(d))])
    plt.figure(figsize=(10,5))
    p = sns.scatterplot(x="HW %", y="Scaled Exam Score", data=d)
    plt.figure(figsize=(10,5))
    p = sns.boxplot(x="HW Percentile by Group", y="Scaled Exam Score", data=d)
    add_median_label_on_boxplot(d, p, "HW Percentile by Group", "Scaled Exam Score", p.get_xticks())

def time_vs_hw_score(data, num_groups, num_entries_threshold):
    d = data.copy()
    d=data.sort_values(by="Median Time Before Deadline")
    d=d[d["Number of Submissions"]>=num_entries_threshold]
    num_per_group = len(d)/num_groups
    d["Group by Submission Time"]=np.floor([i/num_per_group for i in range(len(d))])
    group_maxes = d[["Group by Submission Time", "Median Time Before Deadline"]].groupby(by="Group by Submission Time").max()
    group_mins = d[["Group by Submission Time", "Median Time Before Deadline"]].groupby(by="Group by Submission Time").min()
    
    plt.figure(figsize=(10,5))
    p = sns.boxplot(x="Group by Submission Time", y="HW %", data=d)
    add_median_label_on_boxplot(d, p, "Group by Submission Time", "HW %", p.get_xticks())
    return p, group_maxes, group_mins

def time_vs_hw_views(data, num_groups, num_entries_threshold):
    d=data.copy()
    d=d.sort_values(by="Median Time Before Deadline")
    d = d[d["Number of Submissions"]>=num_entries_threshold]
    num_per_group = len(d)/num_groups
    d["Group by Submission Time"]=np.floor([i/num_per_group for i in range(len(d))])
    
    group_maxes = d[["Group by Submission Time", "Median Time Before Deadline"]].groupby(by="Group by Submission Time").max()
    group_mins = d[["Group by Submission Time", "Median Time Before Deadline"]].groupby(by="Group by Submission Time").min()
    plt.figure(figsize=(10,5))
    p = sns.boxplot(x="Group by Submission Time", y="View Per HW", data=d, showfliers=False)
    add_median_label_on_boxplot(d, p, "Group by Submission Time", "View Per HW", p.get_xticks())
    return p, group_maxes, group_mins

def time_vs_exam_views(data, num_groups, num_entries_threshold):
    d=data.copy()
    d=d.sort_values(by="Median Time Before Deadline")
    d = d[d["Number of Submissions"]>=num_entries_threshold]
    num_per_group = len(d)/num_groups
    d["Group by Submission Time"]=np.floor([i/num_per_group for i in range(len(d))])
    group_maxes = d[["Group by Submission Time", "Median Time Before Deadline"]].groupby(by="Group by Submission Time").max()
    group_mins = d[["Group by Submission Time", "Median Time Before Deadline"]].groupby(by="Group by Submission Time").min()

    plt.figure(figsize=(10,5))
    p = sns.boxplot(x="Group by Submission Time", y="View Per Exam", data=d, showfliers = False)
    add_median_label_on_boxplot(d, p, "Group by Submission Time", "View Per Exam", p.get_xticks())
    return p, group_maxes, group_mins

def num_hw_entries_vs_exam_score(data):
    d=data.copy()
    plt.figure(figsize=(10,5))
    p=sns.boxplot(x='Number of Submissions', y="Scaled Exam Score", data=d)
    x_ticks = np.array(p.get_xticks())+1
    add_median_label_on_boxplot(d, p, "Number of Submissions", "Scaled Exam Score", x_ticks)

def hw_view_vs_exam_views(data): 
    d = data.copy()
    plt.figure(figsize=(10,5))
    p =sns.scatterplot(x="View Per Exam", y="View Per HW", data=d)

def hw_views_vs_hw_scores(data, num_groups, num_entries_threshold):
    d = data.copy()
    d = d[["HW %", "View Per HW", "Number of Submissions"]]
    d=d[d["Number of Submissions"]>=num_entries_threshold]
    
    num_per_group = len(d)/num_groups
    d=d.sort_values(by="HW %")
    d["HW Percentile by Group"]=np.floor([i/num_per_group for i in range(len(d))])
    plt.figure(figsize=(10,5))
    p1 = sns.boxplot(x="HW Percentile by Group", y="View Per HW", data=d)
    add_median_label_on_boxplot(d, p1, "HW Percentile by Group", "View Per HW", p1.get_xticks())
    
    
    plt.figure(figsize=(10,5))
    d["View Per HW Floored"]=np.floor(d["View Per HW"])
    p2 = sns.boxplot(x="View Per HW Floored", y="HW %", data=d)
    
    plt.figure(figsize=(10,5))
    p3 = sns.distplot(d["View Per HW Floored"], kde=False)
    p3.set_ylabel("Number of Students")
    add_median_label_on_boxplot(d, p2, "View Per HW Floored", "HW %", p2.get_xticks())

def gender_vs_exam_scores(data):
    d = data.copy()
    x_ticks = np.sort(["F", "M", "Unknown"])
    plt.figure(figsize=(8,4))
    p1 = sns.boxplot(d["Gender"], d["Scaled Exam Score"], order=x_ticks)
    add_median_label_on_boxplot(d, p1, "Gender", "Scaled Exam Score", x_ticks)
    
    #overlayed histograms
    maximum_score = 78
    minimum_score = 18
    bin_size = 1
    plt.figure(figsize=(8,4))
    p2 = sns.distplot(d[d["Gender"]=="M"]["Scaled Exam Score"], kde=False, label="M", bins=np.arange(minimum_score, maximum_score, bin_size))
    p2 = sns.distplot(d[d["Gender"]=="F"]["Scaled Exam Score"], kde=False, label="F", bins=np.arange(minimum_score, maximum_score, bin_size))
    p2 = sns.distplot(d[d["Gender"]=="Unknown"]["Scaled Exam Score"], kde=False, label="U", bins=np.arange(minimum_score, maximum_score, bin_size))
    p2.set_ylabel("Number of Students")
    plt.legend() 

def gender_vs_hw_scores(data):
    d = data.copy()
    x_ticks = np.sort(["F", "M", "Unknown"])
    plt.figure(figsize=(8,4))
    p1 = sns.boxplot(d["Gender"], d["HW %"], order=x_ticks)
    add_median_label_on_boxplot(d, p1, "Gender", "HW %", x_ticks)
    
    #overlayed histograms
    plt.figure(figsize=(8,4))
    p2 = sns.distplot(d[d["Gender"]=="M"]["HW %"], kde=False, label="M")
    p2 = sns.distplot(d[d["Gender"]=="F"]["HW %"], kde=False, label="F")
    p2 = sns.distplot(d[d["Gender"]=="Unknown"]["HW %"], kde=False, label="U")
    p2.set_ylabel("Number of Students")
    plt.legend() 

def gender_vs_time(data):
    d = data.copy()
   
    #overlayed histograms
    start_time = -1339
    end_time = 1
    bin_size = 60
    plt.figure(figsize=(8,4))
    p = sns.distplot(d[d["Gender"]=="M"]["Median Time Before Deadline"], kde=False, label="M", bins=np.arange(start_time, end_time, bin_size))
    p = sns.distplot(d[d["Gender"]=="F"]["Median Time Before Deadline"], kde=False, label="F", bins=np.arange(start_time, end_time, bin_size))
    p.set_ylabel("Number of Students")
    plt.legend() 

def gender_vs_hw_views(data):
    d = data.copy()
    x_ticks = np.sort(["F", "M", "Unknown"])
    plt.figure(figsize=(8,4))
    p1 = sns.boxplot(d["Gender"], d["View Per HW"], order=x_ticks)
    add_median_label_on_boxplot(d, p1, "Gender", "View Per HW", x_ticks)
    
    #overlayed histograms
    plt.figure(figsize=(8,4))
    p2 = sns.distplot(d[d["Gender"]=="M"]["View Per HW"], kde=False, label="M")
    p2 = sns.distplot(d[d["Gender"]=="F"]["View Per HW"], kde=False, label="F")
    p2 = sns.distplot(d[d["Gender"]=="Unknown"]["View Per HW"], kde=False, label="U")
    p2.set_ylabel("Number of Students")
    plt.legend() 


