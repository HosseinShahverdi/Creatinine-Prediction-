import pandas as pd

def PurifyExaminExcels(path):
  df_temp=pd.read_excel(path)
  df=df_temp[
    ['Age', 'FBS', 'Cholesterol', 'Triglycerides', 'HDL Cholesterol', 'HGB', 'HCT', 'Creatinine']
  ]
  df.Age = df.Age.str.strip(' سال')
  df.dropna(how='any', inplace=True)

  df=df[df.FBS.str.contains('\*')==False]
  df=df[df.Cholesterol.str.contains('\*')==False]
  df=df[df.Triglycerides.str.contains('\*')==False]
  df=df[df.HGB.str.contains('\*')==False]
  df=df[df.HCT.str.contains('\*')==False]
  df=df[df.Creatinine.str.contains('\*')==False]
  df=df[df.Age.str.contains("ماه")==False]

  df.Age=df.Age.astype(int)
  df.FBS=df.FBS.astype(int)
  df.Cholesterol=df.Cholesterol.astype(int)
  df.Triglycerides=df.Triglycerides.astype(float)
  df.HGB=df.HGB.astype(float)
  df.HCT=df.HCT.astype(float)
  df.Creatinine=df.Creatinine.astype(float)

  df=df.sort_values(by="Age", ascending=True)

  return df
  
excel_in_put = "1398.xlsx"
excel_out_put = "1398-.xlsx"

df = PurifyExaminExcels(excel_in_put)
df.to_excel(excel_out_put, index=False)
