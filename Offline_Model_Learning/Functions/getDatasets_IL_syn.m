function [utrain, ytrain, uval, yval, utest, ytest] = getDatasets_IL_syn(dataStruct, ratio_train, ratio_val)

    % Daten aus dem Base-Workspace laden
    % dataStruct = evalin('base', 'dataStruct');

    % Daten aus dataStruct extrahieren
    DOI_main = dataStruct.DOI_main';
    P2M = dataStruct.SOI_pre';
    SOI_main = dataStruct.SOI_main;
    H2_doi = dataStruct.DOI_H2;

    % IMEP Simulation
    IMEP = dataStruct.IMEP_sim; 

    % IMEP Referenzen
    IMEP_ref0 = dataStruct.C2C_NMPC_IMEP_load_ref0;
    IMEP_ref1 = dataStruct.C2C_NMPC_IMEP_load_ref1';
    IMEP_ref2 = dataStruct.C2C_NMPC_IMEP_load_ref2;
    IMEP_old = dataStruct.IMEP_ref';

    % Länge des Datensatzes und Indizes für Training und Testen
    dataset_length = size(dataStruct.IMEP_ref, 1);
    indx_tr = floor(ratio_train * dataset_length);
    indx_tst = floor(ratio_val * dataset_length);

    % Variablen zuweisen für IL
    y1 = DOI_main';
    y2 = P2M';
    y3 = SOI_main;
    y4 = H2_doi;

    u1 = IMEP_ref0;
    u2 = IMEP_ref1';
    u3 = IMEP_ref2;
    u4 = IMEP_old';

    % Trainingdaten
    utrain = [u1(1:indx_tr), u2(1:indx_tr), u3(1:indx_tr), u4(1:indx_tr)];
    ytrain = [y1(1:indx_tr), y2(1:indx_tr), y3(1:indx_tr), y4(1:indx_tr)];

    % Validierungsdaten
    uval = [u1(indx_tr+1:indx_tst), u2(indx_tr+1:indx_tst), u3(indx_tr+1:indx_tst), u4(indx_tr+1:indx_tst)];
    yval = [y1(indx_tr+1:indx_tst), y2(indx_tr+1:indx_tst), y3(indx_tr+1:indx_tst), y4(indx_tr+1:indx_tst)];

    % Testdaten
    utest = [u1(indx_tst+1:end), u2(indx_tst+1:end), u3(indx_tst+1:end), u4(indx_tst+1:end)];
    ytest = [y1(indx_tst+1:end), y2(indx_tst+1:end), y3(indx_tst+1:end), y4(indx_tst+1:end)];

end
