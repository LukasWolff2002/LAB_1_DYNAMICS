function [A,t] = lee_archivo_labs2(archivo)

ffm2=archivo;

fid = fopen(ffm2);

for i=1:100
    tline = fgetl(fid);
%     disp(tline)
    if length(tline)>=11
        if strcmp(tline(1:7),'0.00000')
            break
        end
    end
end


aux=1;

while ~feof(fid)
    
    A1 = sscanf(tline,'%f');
    
    t(aux)=A1(1);
    A(aux,:)=A1([2,4])';
%     tline = fgetl(fid);
    aux=aux+1;
    tline = fgetl(fid);
    %[Tiempo[ms],axA1[g],azA1[g],axA2[g],azA2[g],axA3[g],azA3[g],axB1[g],azB1[g],axB2[g],azB2[g],axB3[g],azB3[g],axC1[g],azC1[g],axC2[g],azC2[g],axC3[g],azC3[g],T° [°C]] 
    
end


end

