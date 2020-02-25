
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

#------------------------------------------------------------------------------#
''' sample dataframe ''' 

df = pd.DataFrame({'A': 1.,
                   'B': pd.date_range('20180101', periods=20),
                   'C': pd.Series(1, index=list(range(20)), dtype='float32'),
                   'D': np.array([3] * 20, dtype='int32'),
                   'E': pd.Categorical(((["test"] * 10) + (["train"] * 10))),
                   'F': pd.Categorical(['foo', 'bar', 'fizz', 'buzz'] * 5), 
                   'G': range(1, 20 + 1), 
                   'H': np.random.randint(0,10, size=20)})

df.head(n=10)

df.info()



###==========================================================================###
### SQL to Pandas Translations  =============================================###
###==========================================================================###


#------------------------------------------------------------------------------#
''' subsetting by rows and columns - use .loc '''

# sql: select <columns> from <table> where <conditions> 
#   the "from <table>" comes from the dataframe name, 
#   the where conditions happen in the row selection parts of .loc[] 
#   the column selections happen in the second part of .loc[]
    
# sql 1: basic select columns with simple where condition
#   select A, B, G from df where B >= '2018-01-10' 
df1 = df.loc[df['B'] >= '2018-01-10', ['A', 'B', 'G']]

df1.head()

# multiple where criteria 
#   use () for each and logical operators 
#   use bitwise operators: & (and), | (or), ~ (not), ^ (xor)
#   note that code within the .loc square brackets can span multiple lines

# sql 2: basic column selection with multiple where conditions 
#   select A, B, E, G from df where B >= '2018-01-10' and E == 'test'
df2 = df.loc[(df['B'] >= '2018-01-10') & (df['E'] == 'test'), 
             ['A', 'B', 'E', 'G']]

df2.head()

#------------------------------------------------------------------------------#
''' subsetting and adding calculated columns ''' 

# sql 3: column selection, multiple where conditions and add new columns 
#   select A, B, E, G, G ** 2 as G_squared, 
#       D ** 3 as D_cubed, A + G as A_plus_G 
#   from df where B >= '2018-01-10' and E == 'test'
# Notes: 
#   enclose entire statement in () to spread over multiple lines 
#   new columns are added with the .assign() method, which allows multiple 
#   column assignments and all columns are available in calculations, even 
#   if not in final dataframe 
df3 = (df.loc[(df['B'] >= '2018-01-10') 
              & (df['E'] == 'test'), 
              ['A', 'B', 'E', 'G']]
       .assign(G_squared = df['G'] ** 2, 
               D_cubed = df['D'] ** 3, 
               A_plus_G = df['A'] + df['G']))


df3.head()


#------------------------------------------------------------------------------#
''' sorting results ''' 
# in general tacking the .sort_values() method on the end of a statement 
# will sort the results 

# sql 4: basic column selection with multiple where conditions and order by  
#   select A, B, E, G from df where B >= '2018-01-10' and E == 'test' 
#   order by G desc 
df4 = df.loc[(df['B'] >= '2018-01-10') & (df['E'] == 'test'), 
             ['A', 'B', 'E', 'G']].sort_values(by='G', ascending = False)

df4.head()

# sql 5: same as sql 4 but order by multiple columns 
#   select A, B, E, G from df where B >= '2018-01-10' and E == 'test' 
#   order by B, G 
# Note: when sorting by multiple values, enter a list in square brackets 
#       ascending argument is also a list of booleans 
df5 = df.loc[(df['B'] >= '2018-01-10') & (df['E'] == 'test'), 
             ['A', 'B', 'E', 'G']
             ].sort_values(by=['B', 'G'], ascending = [True, True])

df5.head()

# sql 6: column selection, multiple where conditions and add new columns 
#           with order by  
#   select A, B, E, G, G ** 2 as G_squared, 
#       D ** 3 as D_cubed, A + G as A_plus_G 
#   from df where B >= '2018-01-10' and E == 'test'
#   order by A_plus_G desc 
df6 = (df.loc[(df['B'] >= '2018-01-10') 
              & (df['E'] == 'test'), 
              ['A', 'B', 'E', 'G']]
       .assign(G_squared = df['G'] ** 2, 
               D_cubed = df['D'] ** 3, 
               A_plus_G = df['A'] + df['G'])
       .sort_values(by='A_plus_G', ascending = False))

df6.head()


#------------------------------------------------------------------------------#
''' aggregating multiple columns and specifying names ''' 
# named aggregation - see pandas docs 
# in the .agg() function include the following: 
# .agg(calculated_col_name1 = ('reference column', 'function'), 
#      calculated_col_name2 = ('reference column', 'function'))
# reset_index is also interesting 
#   useful to reset the index to make filtering easier 
#   add .reset_index() at the end and grouping columns will be included in 
#   the resulting dataset 
#   .reset_index(drop=True) will NOT include the grouping columns in the 
#   resulting dataset 
#   resetting the index also drops the level added in the column names from 
#   the multi-index that gets returned by default 


# sql 7: group by one or more columns, calculate aggregates, and sort 
#   select E, F, min(B) as first_date, sum(G) as G_sum, mean(H) as H_mean 
#   from df group by E, F 

df7 = df.groupby(['E', 'F']).agg(
            first_date = ('B', 'min'), 
            G_sum = ('G', 'sum'), 
            H_mean = ('H', 'mean')).reset_index() 

df7

# sql 8: filter dataset, group by one or more columns, calculate aggregates, 
#       and order by 
#   select E, F, min(B) as first_date, sum(G) as G_sum, mean(H) as H_mean 
#   from df 
#   where F != 'bar' 
#   group by E, F 
#   order by E, first_date 
# Note: filtering is a little odd - can use .filter() method but it requires 
#   passing in a function. A slightly more intuitive way is to filter the 
#   dataframe at the beggining, group on that, and drop the NaNs at the end 
#   also important to highlight that order of operations matters - reset_index
#   should come at the very end, otherwise the index will not be in order 

df8 = (df.loc[df['F'] != 'bar'].groupby(['E', 'F']).agg(
        first_date = ('B', 'min'), 
            G_sum = ('G', 'sum'), 
            H_mean = ('H', 'mean'))
        .sort_values(by=['E', 'first_date'])
        .reset_index()
        .dropna())

df8

#------------------------------------------------------------------------------#
''' left joins '''

# create a smaller dataset to merge onto the main one 
# mimics the common example of a lookup table of flags 
df_right = pd.DataFrame({'A_flags': pd.Categorical(
                                    ['foo', 'bar', 'fizz', 'buzz']), 
                         'F_flags': [101, 202, 303, 404]})

# basic merge, explicit column syntax 
# this brings in all the columns from each dataset 
df_merged1 = (pd.merge(df, df_right, how='left', 
                      left_on = 'F', right_on = 'A_flags'))

df_merged1.head()

# same merge syntax, but drops the common merge column using .drop() 
df_merged2 = (pd.merge(df, df_right, how='left', 
                      left_on = 'F', right_on = 'A_flags')
             .drop('A_flags', axis=1)) 

df_merged2.head()

# same merge syntax but uses .filter() as a "keep" statement for the columns 
#   you want to keep in the final dataset - slightly more explicit a
df_merged3 = (pd.merge(df, df_right, how='left', 
                      left_on = 'F', right_on = 'A_flags')
             .filter(['B', 'F', 'G', 'H', 'F_flags'], 
                     axis=1)) 

df_merged3.head()




#------------------------------------------------------------------------------#
''' additional tasks: 
        group by having 
'''



#------------------------------------------------------------------------------#
''' selecting single value - use .loc[<conditions>].values[0] '''

# get single value into variable 
gtgt = df.loc[df['B'] == '2018-01-02', 'G'].values[0]

# index calculation 
df['G1'] = (df['G'] / gtgt) * 100
 


#------------------------------------------------------------------------------#
''' rule of thumb for column selections - use brackets, not dot notation ''' 

# selecting one column only needs 1 set of brackets 
df['G']

# multiple columns need a list 
df[['A', 'F', 'G']]

# can also select columns with variable names as a list variable
cols = ['A', 'F', 'G']
df2 = df[cols] 

df2.head()


#------------------------------------------------------------------------------#
''' add flag columns based on various criteria ''' 

# simple if/else - use np.where 
# you can nest multiple criteria as long as there are only 2 outcomes 
df['flag1_GH_even'] = np.where((df['G'] % 2 == 0) & (df['H'] % 2 == 0), 
                                  'Both Even', 'Not Even') 

# more complex set of conditions - use np.select 
# np.select takes a list of conditions and maps them to another list of 
#   outcomes, with the final argument being the default for anything else

df['flag2'] = np.select([(df['F'].isin(['foo', 'fizz'])), (df['F'] == 'bar')],
                         ['foo-fizz', 'bar'], 
                         'buzz default') 

# a cleaner method is to put the lists in variables and use those 
# the code below does the same as "flag2" 
# first, put conditions in a list using dataframe specific syntax 
# next, put the responses in another list (in same order) 
# include a default value in the np.select function 

conditions = [(df['F'].isin(['foo', 'fizz'])), (df['F'] == 'bar')]

choices = ['foo-fizz', 'bar']

df['flag3'] = np.select(conditions, choices, 'buzz default')

df.head(n=10)



df.head()


###==========================================================================###
### Plotting notes  =========================================================###
###==========================================================================###


#------------------------------------------------------------------------------#
''' line plots '''


# line plot time series data 

dfts = pd.DataFrame({'dt': pd.date_range('20180101', periods=252),
                     'rand': np.random.normal(0,0.5,size=252), 
                     'updrift': (np.linspace(0,2,252) 
                                 + np.random.normal(0,0.25,252)) })

# quick plots with pandas using the plot() function 
dfts.plot(x='dt', y=['rand', 'updrift'])

dfts.plot(x='dt', y='rand')


