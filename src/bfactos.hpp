#include <vector>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <vector>
//#ifdef ROOT
//#include "Rootgrapher.h"
//#endif

using namespace std;

void Proteina::flexibility(char *file){
int at;
vector<float> promedios;
vector<float> stds;
printf("atomos en proteina en total %d \n",No_atomos_en_prot );
//.area_atomo
//.No_Atomos2
int ww=0;
for(int l=0; l< cadena;l++){
if(l!=0){
  ww=cadens[l]-cadens[l-1];
}else{
ww=cadens[l]-1;
}
No_atomos_por_cadena.push_back(ww);
}
vector<vector<float> > bf(cadena);
vector<float> surfaces;
//CA

int po=0;
int ress=0;
int identificador=0;
for (int k=0; k <  Lista_Cajas.size(); ++k){
po= Cajas[k].No_atoms;
//    p=Lista_residuos[i].No_atomos;
  for (int kk = 0; kk < po; ++kk){
  //  suma=(Cajas[k].atomos_caja[kk].area_atomo)+suma;
    ress=Cajas[k].atomos_caja[kk].Id_residuo;
  //  At1=Cajas[k].atomos_caja[kk].No_Atomos2;
    identificador=Cajas[k].atomos_caja[kk].No_Atomo;
    for(int kkk=0; kkk< No_residuos;++kkk){
    if(  residuos_en_proteina[kkk].Id_Res==ress){
        for(int at=0; at<residuos_en_proteina[kkk].No_atomos ;++at){
  if( residuos_en_proteina[kkk].atomos_en_residuo[at].No_Atomo ==identificador){
residuos_en_proteina[kkk].atomos_en_residuo[at].area_atomo=Cajas[k].atomos_caja[kk].area_atomo;
break;
  }
}

break;
}
}
}
}

int cc=0;
printf("sali \n");
for(int w=0; w< No_residuos;w++){
 cc=residuos_en_proteina[w].No_atomos;
for(int a=0; a< cc;a++){
surfaces.push_back(residuos_en_proteina[w].atomos_en_residuo[a].area_atomo);

}
}

//bfactor  y asa
ofstream outfile ("infoporatomo.txt");
outfile << "N   "<<file<<endl;
outfile << "A   "<<cadena<< endl;
for(int v=0; v< cadena;v++){
outfile << "B   "<<No_atomos_por_cadena[v]<< endl;
}
float carry;
float area;
int cont=0;
float mean;
for(int i=0; i< cadena;i++){
mean=0;
  int prue=No_atomos_por_cadena[i];
for(int j=0; j<prue ;j++){
carry=factores[cont];
area=surfaces[cont];
outfile <<i<<"   "<<carry<<"                "<<area<<endl;
mean=mean+carry;
cont++;
bf[i].push_back(carry);
}
promedios.push_back(mean);
}

float local=0;
float stand=0;
for(int k=0; k< cadena;k++){
  local=promedios[k]/No_atomos_por_cadena[k];
bmean.push_back(local);
}
outfile.close();

float std_local=0;
for(int it=0; it< cadena;it++){
std_local=std(bmean[it],bf[it] );
stds.push_back(std_local);
}


float st=0;
float prom=0;
for(int n=0; n< No_residuos ;n++){
for(int g=0; g< cadena;g++){
if((g+1) == residuos_en_proteina[n].id_cadena ){
st=stds[g];
prom=bmean[g];
break;
}
}
residuos_en_proteina[n].b_factor=((residuos_en_proteina[n].b_carbono_central-prom)/st );
}


for(int r=0; r< cadena;r++){
printf("promedio bfactor en el id cadena %d : %f \n",r,bmean[r] );
printf("desv estandar de bfacor en id cadena %d : %f \n",r,stds[r] );
}


}
