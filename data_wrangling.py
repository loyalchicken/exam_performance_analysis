import pandas as pd 
import numpy as np
import pytz

def number_of_homework_submissions_by_student(merged_hws, num_hws):
  return pd.Series([num_hws - int(merged_hws.iloc[i].isnull().sum() / 3) for i in range(len(merged_hws))], name="number of submissions")

def overall_hw_grade_by_student(merged_hws, num_hws):
  hw_scores_titles = ['Total Score, HW %d' % i for i in range(num_hws)]
  hw_scores = merged_hws[hw_scores_titles]
  overall_hw_grade = pd.Series((hw_scores.sum(axis=1) / merged_hws["Number of Submissions"]), name="overall percentage")
  return overall_hw_grade

def hw_views_by_student(merged_hws, num_hws):
  hw_views_titles = ['View Count, HW %d' % i for i in range(num_hws)]
  hw_views = merged_hws[hw_views_titles]
  hw_views = pd.Series((hw_views.sum(axis=1) / merged_hws["Number of Submissions"]).dropna(), name="average number of views per hw")
  return hw_views

def hw_submission_time_before_deadline_by_student(merged_hws, deadlines, num_hws):
  submission_time_titles = ['Submission Time, HW %d' % i for i in range(num_hws)]
  submission_times = merged_hws[submission_time_titles]

  #don't consider hw0 since it's an outlier (see histograms in 'DATA EXPLORATION: STATS BY HOMEWORK' section below)
  homeworks_to_consider = np.arange(1,num_hws)
  submission_times_before_deadline = pd.DataFrame(np.array([hw_submission_times(i, submission_times, deadlines) for i in homeworks_to_consider]).T)
  mean_submission_times_before_deadline_by_student = pd.Series(submission_times_before_deadline.sum(axis=1)/merged_hws["Number of Submissions"])
  median_submission_times_before_deadline_by_student = submission_times_before_deadline.median(axis=1).dropna()
  
  return median_submission_times_before_deadline_by_student, mean_submission_times_before_deadline_by_student

def hw_submission_times(n, submission_times, deadlines):
    """Returns a series of homework submission times before deadline (in minutes).
        Series includes null values 

    Parameters
    ----------
    n                      integer (homework #)
    submission_times       dataframe with submission times for students of interest  

    Returns
    -------
    A Series of length len(submission_times)
    """

    if n == 5:
        times_datetime64_format = [pd.to_datetime(submission_times.iloc[:, n][i]) for i in range(len(submission_times))] #manually applies pd.date_time since this series contains entries with multiple timezones
        times = pd.Series([np.NaN if type(times_datetime64_format[i])==pd._libs.tslibs.nattype.NaTType else (times_datetime64_format[i]-deadlines[n][1] if times_datetime64_format[i].tz==pytz.FixedOffset(-420) else times_datetime64_format[i]-deadlines[n][0]) for i in range(len(times_datetime64_format))])
    else:
        times = pd.to_datetime(submission_times.iloc[:,n]) - deadlines[n]
    return pd.Series(times.apply(lambda x: x.total_seconds()/60), name="submission time before deadline (minutes)")

def exam_views_by_student(merged_hws, exams):
  return exams.merge(merged_hws[["SID"]], on="SID", how='inner')[["SID", "View Count M1", "View Count M2", "View Count Final", "View Count Exam Total", "View Per Exam"]]

def exam_scores_by_student(merged_hws, exams):
  return exams.merge(merged_hws[["SID"]], on="SID", how='inner')[["SID", "Scaled Exam Score"]]

def gender_by_student():
  #get genders
  file1 = open('data/gender.txt', 'r') 
  Lines = file1.readlines() 
  count = 0
  students = []
  for line in Lines: 
      students.append(line.split(","))

  genders = np.array(students).astype(float).astype(int)

  def number_to_gender(n):
      if n == 0:
          return "M"
      elif n == 1:
          return "F"
      return "Unknown"

  genders = pd.DataFrame({'SID': pd.Series(genders[:, 0]).apply(str), 'Gender': pd.Series(genders[:, 1]).apply(number_to_gender)})
  return genders

