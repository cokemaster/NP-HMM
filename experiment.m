function experiment
    %load('./MarketData/spread_dbid_dask_data.mat'); %O, Nm, values
    %load('./MarketData/spread_data.mat'); %O, Nm, values
    T = 100;
    [O, Nm] = generateArtificialData(T);
    K = 2;
    num_iter = 20;
    train_hmm_em(O, Nm, K, num_iter, 1)
end