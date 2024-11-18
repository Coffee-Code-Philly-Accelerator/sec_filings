import json
import numpy as np

def exceptions()->dict:
    return {
        '3906/2004-12-31/Schedule_of_Investments_2.csv':{
            'portfolio_company':[True]+[False]*24,
            'investment':[False,True]+[False]*23,
            'interest':[False]*3+[True]+[False]*21,
            'other':[False]*4+[True]+[False]*20,
            'value_prev':[False]*6+[True]+[False]*18,
            'additions':[False]*9+[True]+[False]*15,
            'reductions':[False]*12+[True]+[False]*12,
            'value_cur':[False]*15+[True]+[False]*9,
        },
        '3906/2004-12-31/Schedule_of_Investments_0.csv':{
            'portfolio_company':[True]+[False]*23,
            'investment':[False,True]+[False]*22,
            'interest':[False]*3+[True]+[False]*20,
            'other':[False]*4+[True]+[False]*19,
            'value_prev':[False]*6+[True]+[False]*17,
            'additions':[False]*9+[True]+[False]*14,
            'reductions':[False]*12+[True]+[False]*11,
            'value_cur':[False]*15+[True]+[False]*8,
        },
        '3906/2004-12-31/Schedule_of_Investments_1.csv':{
            'portfolio_company':[True]+[False]*23,
            'investment':[False,True]+[False]*22,
            'interest':[False]*3+[True]+[False]*20,
            'other':[False]*4+[True]+[False]*19,
            'value_prev':[False]*6+[True]+[False]*17,
            'additions':[False]*9+[True]+[False]*14,
            'reductions':[False]*12+[True]+[False]*11,
            'value_cur':[False]*15+[True]+[False]*8,
        },
        '3906/2001-09-30/Schedule_of_Investments_0.csv':{
            'portfolio_company':[True]+[False]*11,
            'investment':[False]*2+[True]+[False]*9,
            'cost':[False]*4+[True]+[False]*7,
            'value':[False]*7+[True]+[False]*4,
        },
        '3906/2001-09-30/Schedule_of_Investments_2.csv':{
            'portfolio_company':[True]+[False]*11,
            'investment':[False]*2+[True]+[False]*9,
            'cost':[False]*4+[True]+[False]*7,
            'value':[False]*7+[True]+[False]*4,
        },
        '3906/2001-09-30/Schedule_of_Investments_4.csv':{
            'portfolio_company':[True]+[False]*11,
            'investment':[False]*2+[True]+[False]*9,
            'cost':[False]*4+[True]+[False]*7,
            'value':[False]*7+[True]+[False]*4,
        },
        '3906/2001-09-30/Schedule_of_Investments_6.csv':{
            'portfolio_company':[True]+[False]*11,
            'investment':[False]*2+[True]+[False]*9,
            'cost':[False]*4+[True]+[False]*7,
            'value':[False]*7+[True]+[False]*4,
        },
        '3906/2001-09-30/Schedule_of_Investments_8.csv':{
            'portfolio_company':[True]+[False]*11,
            'investment':[False]*2+[True]+[False]*9,
            'cost':[False]*4+[True]+[False]*7,
            'value':[False]*7+[True]+[False]*4,
        },
        '3906/2001-09-30/Schedule_of_Investments_10.csv':{
            'portfolio_company':[True]+[False]*11,
            'investment':[False]*2+[True]+[False]*9,
            'cost':[False]*4+[True]+[False]*7,
            'value':[False]*7+[True]+[False]*4,
        },
        '3906/2001-09-30/Schedule_of_Investments_12.csv':{
            'portfolio_company':[True]+[False]*10,
            'investment':[False]+[True]+[False]*9,
            'cost':[False]*3+[True]+[False]*7,
            'value':[False]*6+[True]+[False]*4,
        },
    }

if __name__ == "__main__":
    import json
    mask = exceptions()
    with open("manual_mask.json","w") as f:
        json.dump(mask,f,indent=4)