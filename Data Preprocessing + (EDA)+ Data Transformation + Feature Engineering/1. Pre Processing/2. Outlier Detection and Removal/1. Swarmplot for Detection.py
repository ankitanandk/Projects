    '''
    SWARMPLOT: Box plot plus the values plotted together is swarmplot
    '''
    import pandas as pd
    import seaborn as sb
    df=pd.read_csv('Data.csv')
    
    '''
    #Making Box plot for Age, the x and y is just for the orientation of the Plot
    '''
    sb.boxplot(x=df["values"])
    sb.swarmplot(x='values',data=df ,color='red', )
    
    
    '''
    Y i sjust for oreintation of the BOX plot: its a univariate plot
    
    '''
    sb.boxplot(y=df["values"])
    sb.swarmplot(y='values',data=df ,color='red', )
    