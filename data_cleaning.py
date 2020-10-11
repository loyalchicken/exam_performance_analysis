import pandas as pd

def clean_and_merge_exam_data(df_first_exam, df_second_exam, df_final_exam, first_midterm_total_points, second_midterm_total_points, final_total_points, first_midterm_weight, second_midterm_weight, final_weight):
    first=df_first_exam[['SID', 'Total Score', 'View Count']]
    second=df_second_exam[['SID', 'Total Score', 'View Count']]
    final=df_final_exam[['SID', 'Total Score', 'View Count']]
    
    #only include students who wrote all three exams
    exams=first.merge(second, left_on='SID', right_on='SID').dropna()
    exams=exams[['SID', 'Total Score_x', 'Total Score_y', 'View Count_x', 'View Count_y']]
    exams=exams.rename(index=str, columns={"Total Score_x":"M1", "Total Score_y":"M2", "View Count_x":"View Count M1", "View Count_y":"View Count M2"})
    exams=exams.merge(final, left_on='SID', right_on='SID').dropna()
    exams=exams[['SID', 'M1', 'M2', 'Total Score', 'View Count M1', 'View Count M2', 'View Count']]
    exams=exams.rename(index=str, columns={'Total Score':'Final', "View Count":"View Count Final"})
    exams['View Count Exam Total']=exams['View Count M1']+exams['View Count M2']+exams['View Count Final']
    exams['View Per Exam']=exams['View Count Exam Total']/3
    exams['Scaled Exam Score']=exams['M1']/first_midterm_total_points*first_midterm_weight+exams['M2']/second_midterm_total_points*second_midterm_weight+exams['Final']/final_total_points*final_weight
    return exams


def clean_and_merge_homework_data(hws, hw_total_scores, exams):
    #drop rows with null SIDs in each df, then add "Total Score" which is hw grade in %
    for i in range(len(hws)):
        hws[i].dropna(subset=['SID'], how='all', inplace=True)
        hws[i]['Total Score']=hws[i]['Total Score']/hw_total_scores[i]
        hws[i] = hws[i][['SID', 'Total Score', 'View Count', 'Submission Time']]
        hws[i] = hws[i].rename(columns={"Total Score": 'Total Score, HW %d' % i, "View Count": 'View Count, HW %d' % i, "Submission Time": 'Submission Time, HW %d' % i})

    #merge
    from functools import reduce
    merged_hws = reduce(lambda left, right: pd.merge(left, right, on='SID', how='inner'), hws) 
    
    #DROP STUDENTS WHO DIDN'T TAKE ALL THREE EXAMS
    merged_hws_dropped = merged_hws.merge(exams[["SID"]], how="inner", on="SID")
    return merged_hws_dropped