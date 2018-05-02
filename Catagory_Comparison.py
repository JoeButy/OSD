import pandas as pd
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
# from scipy.stats.stats import pearsonr

### initialize and settings
start_time = time.time()
print '\n\n\t', 'Start Time: ', start_time, '\n'
# Set ipython's max row display to
pd.set_option('display.max_row', 10)
# Set iPython's max column width to:
pd.set_option('display.max_columns', 12)

### locate files
#change this path to the location with the data files
mypath = r'/users/joebuty/documents/joe/OSD/sample_data/'

#Nomenclature
grd = 'ExportGrade'
reading = 'Reading'
math = 'Mathematics'
sid = 'StudentId'
xid = 'external_identifier'
ui = 'usage_index'
sub = 'subject'
school = 'SchoolName'

rd_df = pd.read_csv(mypath+'performance_challenge_reading_data.csv')
ma_df = pd.read_csv(mypath+'performance_challenge_math_data.csv')
ui_df = pd.read_csv(mypath+'performance_challenge_OSD_data.csv')

# ASSUPTION: Student ID and External Identifier are equivalent.
ui_df = ui_df.dropna()
ui_df[sid] = ui_df[xid].astype('int64').astype(str)
ui_df = ui_df.drop([xid], axis=1)

read_scores = rd_df
math_scores = ma_df
#pull top usage index per student id
# TODO throw list of students with more than 1 Usage index?
usage_index = ui_df.groupby([sid, 'subject']).max().reset_index()

# Only consider the best score for a student in a particular subject.
not_group = ['PR', 'DateTaken', 'Characteristics', 'IRL', 'Ethnicity', ui]
read_group = list(read_scores)
read_group = [e for e in read_group if e not in not_group]
read_scores = read_scores.groupby(read_group).max()['PR'].reset_index()
math_group = list(math_scores)
math_group = [e for e in math_group if e not in not_group]
math_scores = math_scores.groupby(math_group).max()['PR'].reset_index()
# add subject for scores concatenation.
math_scores[sub] = math
read_scores[sub] = reading
all_scores = pd.concat([read_scores, math_scores]).reset_index(drop=True)
# join the scores with the usage index. Now the top score for each student is matched
# with the top usage index for each student ID.
all_scores = pd.merge(all_scores, usage_index, on=[sid, sub], how='left')

#format school names
all_scores[school] = all_scores[school].str.split(' ', 1).str[0]

# Graph Formats, stacked bar colors and meanings.
vlo = 'Very Low'
cvlo = (153, 153, 153)
lo = 'Low'
clo = (204, 204, 204)
hi = 'High'
chi = (26, 255, 26)
vhi = 'Very High'
cvhi = (0, 153, 0)
bin_labels = [vlo, lo, hi, vhi]
colors = [cvlo, clo, chi, cvhi]
def pyrgb(tup):
	divisor = 255.0
	rgb = map(lambda x: x/divisor, tup)
	return rgb
colors = [pyrgb(x) for x in colors]


all_scores.ix[:,grd] = all_scores[grd].astype(str)
#the number of students per grade in each bin.
number_bins = 4
sbin = 'Percentage Bin'
all_scores[sbin] = pd.cut(all_scores['PR'], number_bins, labels=bin_labels)
all_OSD_scores = all_scores[np.isfinite(all_scores[ui])].reset_index(drop=True)
all_non_OSD_scores = all_scores[all_scores[ui].isnull()].reset_index(drop=True)
# Break the data apart by categories of intrest, school, grade, characteristics, etc.
by_grade = [grd, sbin, sid, sub]
by_school = [sbin, sid, sub, school]
by_grade_and_school = [sbin, sid, sub, school, grd]

def reset_idx_cat(df):
	df.columns = df.columns.astype('str')
	return df.reset_index()

def normalize_data(df, norm, subj, x):
	# for certain data sets the total number of students is not
	# as important as the percentage of students. This function will
	# change a count of student belonging to group x (grade, school, etc.)
	if norm:
		# normalize the OSD student on the total students
		totals = norm[subj].sum(axis=1).to_frame()
		totals.columns = ['totals']
		totals = reset_idx_cat(totals)
		df = reset_idx_cat(df)
		df = pd.merge(df, totals, on=x, how='inner')
		df = df.set_index(x, drop=True)
		for pr_bin in df:
			df[pr_bin] = df[pr_bin] / df['totals']
		df = df.drop(['totals'], axis=1)
	return df

# Create subject specific count of score bins by variable.
def split_sub(by_x, detail_level=None, all_students_norm=False):
	# this bit is not super resusable. TODO create object classes to handle the children.
	group_list = [sbin, sub]
	if isinstance(detail_level, list):
		group_list.extend(detail_level)
	elif isinstance(detail_level, str):
		group_list.append(detail_level)
	else:
		pass
	by_x = by_x.groupby(group_list).count()
	by_x = by_x.unstack(0)
	by_x = by_x.dropna(how='all')
# 	print by_x, '\n'*3
	if detail_level is not None:
		math_by_x = by_x.xs(math, level=sub, drop_level=True)
		math_by_x.columns = math_by_x.columns.droplevel()
		math_by_x.columns.name = None
		math_by_x = normalize_data(math_by_x, all_students_norm, math, detail_level)
		read_by_x = by_x.xs(reading, level=sub, drop_level=True)
		read_by_x.columns = read_by_x.columns.droplevel()
		read_by_x.columns.name = None
		read_by_x = normalize_data(read_by_x, all_students_norm, reading, detail_level)
	else:
		math_by_x = by_x.loc[math,:].unstack()#.reset_index(drop=True)
		math_by_x.index = ['Math']

		read_by_x = by_x.loc[reading,:].unstack().reset_index(drop=True)
		read_by_x.index = ['Reading']
	return {math: math_by_x, reading: read_by_x}

# count PR bins 
# all grades and schools -- grouped on Percentile Bin and subject
print list(all_OSD_scores)
print 'split-sub'
by_subj_OSD = split_sub(all_OSD_scores[[sid, sbin, sub]])
by_subj_all = split_sub(all_scores[[sid, sbin, sub]])

# break grade for each school
by_grd_sch_OSD = split_sub(all_OSD_scores[by_grade_and_school], [school, grd])
by_grd_sch_all = split_sub(all_scores[by_grade_and_school], [school, grd])

# break by grades only
by_grade_all = split_sub(all_scores[by_grade], grd)
by_grade_OSD = split_sub(all_OSD_scores[by_grade], grd, by_grade_all)

# break by schools only
by_school_all = split_sub(all_scores[by_school], school)
by_school_OSD = split_sub(all_OSD_scores[by_school], school, by_school_all)

def normalize_all_df(df_dict):
	for subj in df_dict:
		df = df_dict[subj]
		df = df.div(df.sum(axis=1), axis=0)
		df_dict[subj] = df
	return df_dict

by_grade_all = normalize_all_df(by_grade_all)
by_school_all = normalize_all_df(by_school_all)

def set_intersec(ls1, ls2):
	return list(set(ls1).intersection(ls2))

# Slice dataframe to so only comparable sets remain.
def comparable_only(e_df_dict, df_dict):
	for el in list(e_df_dict.keys()):
		ls_1 = list(e_df_dict[el].index)
		ls_2 = list(df_dict[el].index)
		idx = set_intersec(list(e_df_dict[el].index), list(df_dict[el].index))
		e_df_dict[el] = e_df_dict[el].loc[idx]
		df_dict[el] = df_dict[el].loc[idx]
	return e_df_dict, df_dict

by_grade_OSD, by_grade_all = comparable_only(by_grade_OSD, by_grade_all)
by_school_OSD, by_school_all = comparable_only(by_school_OSD, by_school_all)
schools_dict_OSD = {}
schools_dict_all = {}

all_schools_list = list(set().union(by_grd_sch_all[math].index.get_level_values(school), \
		by_grd_sch_all[reading].index.get_level_values(school)))

OSD = {}
all = {}
for scho in all_schools_list:
	for subj in [math, reading]:
		OSD[subj] = by_grd_sch_OSD[subj].loc[scho]
		all[subj] = by_grd_sch_OSD[subj].loc[scho]
	schools_dict_OSD[scho], schools_dict_all[scho] = comparable_only(OSD, all)

def graph_spark_subjects(e_df, df, title='', force_norm=False):
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
	e_df[math].plot.bar(stacked=True, ax=axes[0,0], color=colors)
	axes[0,0].set_title('OSD '+math)
	axes[0,0].legend_.remove()
	e_df[reading].plot.bar(stacked=True, ax=axes[0,1], color=colors)
	axes[0,1].set_title('OSD '+reading)
	axes[0,1].legend_.remove()
	df[math].plot.bar(stacked=True, ax=axes[1,0], color=colors)
	axes[1,0].set_title('All '+math)
	axes[1,0].legend_.remove()
	df[reading].plot.bar(stacked=True, ax=axes[1,1], color=colors)
	axes[1,1].set_title('All '+reading)
	axes[1,1].legend_.remove()
	if force_norm:
		axes[0,0].set_ylim([0, 1])
		axes[0,1].set_ylim([0, 1])
		axes[1,0].set_ylim([0, 1])
		axes[1,1].set_ylim([0, 1])				
	handles, labels = axes[0,0].get_legend_handles_labels()
	fig.legend(labels=labels, loc='lower left', ncol=len(labels), mode='expand')
	fig.subplots_adjust(bottom=0.15)
	fig.subplots_adjust(hspace=0.6)
	fig.savefig(title)
	return fig, axes

# Spectial treatment for only all students plot. 
sub_fig, sub_ax = graph_spark_subjects(by_subj_OSD, by_subj_all)

grade_title = 'Grade Level Precentage Ranking Quartiles of OSD Students'
grade_fig, grade_ax = graph_spark_subjects(by_grade_OSD, by_grade_all, \
	grade_title, True)
school_title = 'Schools Precentage Ranking Quartiles of OSD Students'
school_fig, school_ax = graph_spark_subjects(by_school_OSD, by_school_all, \
	school_title, True)

# school_fig.autofmt_xdate() #this has not been working well. 

print '\n' * 3
all_school_df = {}
OSD_school_df = {}

for scho in all_schools_list:
	print scho
	OSD_grades = {}
	all_grades = {}
	for subj in [math, reading]:
		try:
			OSD_grades[subj] = by_grd_sch_OSD[subj].loc[scho].dropna(how='all')
			all_grades[subj] = by_grd_sch_all[subj].loc[scho].dropna(how='all')
		except KeyError:
			OSD_grades[subj] = pd.DataFrame()
			all_grades[subj] = pd.DataFrame()
	all_grade, OSD_grades = comparable_only(all_grades, OSD_grades)
	all_school_df[scho] = all_grades
	OSD_school_df[scho] = OSD_grades 
	school_graph_title = scho + ' ' + grade_title
	school_grades_fig, school_grades_ax = graph_spark_subjects( \
		OSD_school_df[scho], all_school_df[scho])


#### HERE is the significance analysis. 
#### First
# H0 === Students that have any record of using OSD have a Percentile ranking mean 
#	that is equal to the mean of students that have no record of OSD usage.
# print stats.ttest_ind(a=all_OSD_scores['PR'], b=all_non_OSD_scores['PR'])
OSD_non_OSD_p = stats.ttest_ind(a=all_OSD_scores['PR'], b=all_non_OSD_scores['PR'])
print '\n' * 2
print 'P Value, Difference in OSD and Non-OSD Percentile Ranking:'
print str((1 - OSD_non_OSD_p[1])*100) + '\n'*3


#### Second 
# H0 === All of the usage index have the same PR.
#### This is not going to be solved with an ANOVA since we have non parametric data
##### That is to say that the data is not normally distributed.

unique_ui_values = list(all_OSD_scores[ui].unique())
unique_ui_values.sort()
box_arr_ls = []
is_normal = []
ui_hist_data = all_OSD_scores[ui].astype(int).value_counts()
n_ui_vals = len(all_OSD_scores[ui].astype(int).value_counts().index)
for i in unique_ui_values:
	specific_ui_arr = all_OSD_scores[all_OSD_scores[ui] == i].as_matrix(['PR'])
	box_arr_ls.append(specific_ui_arr)

box_data = all_OSD_scores[[ui, 'PR']]

ui_plot = plt.figure(figsize=(5,5))
ui_box_plot = ui_plot.add_subplot(3,1,1).boxplot(box_arr_ls)

ui_histogram = ui_plot.add_subplot(3,1,2).hist(all_OSD_scores[ui].astype(int), \
	bins=np.arange(n_ui_vals + 1) + 0.5)
ui_violins = ui_plot.add_subplot(3,1,3).violinplot(box_arr_ls, showmeans=False, \
	showmedians=True, showextrema=True)

plt.show()


