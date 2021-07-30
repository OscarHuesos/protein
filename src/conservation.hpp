#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <strstream>
#include <fstream>

using namespace std;

void Proteina::scores(char *file){

int r;
char f[]="XXXXFHY.pdb";
float sc;

for(int w=0; w< cadena;w++){
  try{
    char str[5];
    str[0]='p';
    str[1]='d';
    str[2]='b';
    str[3]='s';
    str[4]='/';
    char fail[5];
    fail[0]='f';
    fail[1]='a';
    fail[2]='i';
    fail[3]='l';
    fail[4]='_';
  printf("%d atomos en cadena %d \n", cadens[w],w);
strcpy (f,file);
r=w+1;
char c;
c=r+48;
printf("%c y c \n",c );
printf("%s file ahora es \n",f );
f[6]=c;
f[7]=46;
f[8]=112;
f[9]=100;
f[10]=98;
//f[11]=32;
printf("%s f despues de c \n",f );
int ww;
if(w!=0){
  ww=res_chain[w]-res_chain[w-1];
}else{
ww=res_chain[w];
}
printf("%d res chan con tantos residuos : %d y diff %d \n",w,res_chain[w],ww);

ifstream Fin;
Fin.open(f);
if(!Fin)throw strcat(fail,f);
string buffer;
string res;
int ress=0;
int ff;
while (getline(Fin, buffer)){
//if(    (buffer.substr(0,6)=="ATOM  ")  && (buffer.substr(21,1)[0]==chain[w]  )    ){
if(buffer.substr(0,4)=="ATOM"){
if(buffer.substr(21,1)[0]==chain[w] ){

ff=atoi(buffer.substr(22,5).c_str());
if(ress!=ff){
sc=atof(buffer.substr(60,6).c_str());
for(int i=0; i< No_residuos;i++){
if(residuos_en_proteina[i].Id_Res==ff){
residuos_en_proteina[i].conservation_score=sc;
break;
}
}
}

ress=ff;
}
}
}
 Fin.close();
}
 catch(char* pMsg) { cerr << endl << "Exception:" << pMsg << endl; }
}

}
