/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) drone_ode_gnsf_get_matrices_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s1[135] = {11, 11, 0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s2[51] = {11, 4, 0, 11, 22, 33, 44, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s3[87] = {11, 7, 0, 11, 22, 33, 44, 55, 66, 77, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s4[3] = {11, 0, 0};
static const casadi_int casadi_s5[7] = {0, 4, 0, 0, 0, 0, 0};
static const casadi_int casadi_s6[45] = {6, 6, 0, 6, 12, 18, 24, 30, 36, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s7[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s8[31] = {6, 4, 0, 6, 12, 18, 24, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s9[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s10[4] = {0, 1, 0, 0};
static const casadi_int casadi_s11[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};

/* drone_ode_gnsf_get_matrices_fun:(i0)->(o0[11x11],o1[11x4],o2[11x7],o3[11x11],o4[11x11],o5[11x11],o6[11x0],o7[0x4],o8[6x6],o9[11],o10[6x6],o11[6x4],o12,o13,o14[17],o15[0],o16[6]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2;
  a0=0.;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  if (res[0]!=0) res[0][2]=a0;
  if (res[0]!=0) res[0][3]=a0;
  if (res[0]!=0) res[0][4]=a0;
  if (res[0]!=0) res[0][5]=a0;
  if (res[0]!=0) res[0][6]=a0;
  if (res[0]!=0) res[0][7]=a0;
  if (res[0]!=0) res[0][8]=a0;
  if (res[0]!=0) res[0][9]=a0;
  if (res[0]!=0) res[0][10]=a0;
  if (res[0]!=0) res[0][11]=a0;
  if (res[0]!=0) res[0][12]=a0;
  if (res[0]!=0) res[0][13]=a0;
  if (res[0]!=0) res[0][14]=a0;
  if (res[0]!=0) res[0][15]=a0;
  if (res[0]!=0) res[0][16]=a0;
  if (res[0]!=0) res[0][17]=a0;
  if (res[0]!=0) res[0][18]=a0;
  if (res[0]!=0) res[0][19]=a0;
  if (res[0]!=0) res[0][20]=a0;
  if (res[0]!=0) res[0][21]=a0;
  if (res[0]!=0) res[0][22]=a0;
  if (res[0]!=0) res[0][23]=a0;
  if (res[0]!=0) res[0][24]=a0;
  if (res[0]!=0) res[0][25]=a0;
  if (res[0]!=0) res[0][26]=a0;
  if (res[0]!=0) res[0][27]=a0;
  if (res[0]!=0) res[0][28]=a0;
  if (res[0]!=0) res[0][29]=a0;
  if (res[0]!=0) res[0][30]=a0;
  if (res[0]!=0) res[0][31]=a0;
  if (res[0]!=0) res[0][32]=a0;
  if (res[0]!=0) res[0][33]=a0;
  if (res[0]!=0) res[0][34]=a0;
  if (res[0]!=0) res[0][35]=a0;
  if (res[0]!=0) res[0][36]=a0;
  if (res[0]!=0) res[0][37]=a0;
  if (res[0]!=0) res[0][38]=a0;
  if (res[0]!=0) res[0][39]=a0;
  if (res[0]!=0) res[0][40]=a0;
  if (res[0]!=0) res[0][41]=a0;
  if (res[0]!=0) res[0][42]=a0;
  if (res[0]!=0) res[0][43]=a0;
  if (res[0]!=0) res[0][44]=a0;
  if (res[0]!=0) res[0][45]=a0;
  if (res[0]!=0) res[0][46]=a0;
  if (res[0]!=0) res[0][47]=a0;
  if (res[0]!=0) res[0][48]=a0;
  if (res[0]!=0) res[0][49]=a0;
  if (res[0]!=0) res[0][50]=a0;
  if (res[0]!=0) res[0][51]=a0;
  if (res[0]!=0) res[0][52]=a0;
  if (res[0]!=0) res[0][53]=a0;
  if (res[0]!=0) res[0][54]=a0;
  if (res[0]!=0) res[0][55]=a0;
  if (res[0]!=0) res[0][56]=a0;
  if (res[0]!=0) res[0][57]=a0;
  if (res[0]!=0) res[0][58]=a0;
  if (res[0]!=0) res[0][59]=a0;
  if (res[0]!=0) res[0][60]=a0;
  if (res[0]!=0) res[0][61]=a0;
  if (res[0]!=0) res[0][62]=a0;
  if (res[0]!=0) res[0][63]=a0;
  if (res[0]!=0) res[0][64]=a0;
  if (res[0]!=0) res[0][65]=a0;
  if (res[0]!=0) res[0][66]=a0;
  if (res[0]!=0) res[0][67]=a0;
  if (res[0]!=0) res[0][68]=a0;
  if (res[0]!=0) res[0][69]=a0;
  if (res[0]!=0) res[0][70]=a0;
  if (res[0]!=0) res[0][71]=a0;
  if (res[0]!=0) res[0][72]=a0;
  if (res[0]!=0) res[0][73]=a0;
  if (res[0]!=0) res[0][74]=a0;
  if (res[0]!=0) res[0][75]=a0;
  if (res[0]!=0) res[0][76]=a0;
  if (res[0]!=0) res[0][77]=a0;
  if (res[0]!=0) res[0][78]=a0;
  if (res[0]!=0) res[0][79]=a0;
  if (res[0]!=0) res[0][80]=a0;
  if (res[0]!=0) res[0][81]=a0;
  if (res[0]!=0) res[0][82]=a0;
  if (res[0]!=0) res[0][83]=a0;
  a1=25.;
  if (res[0]!=0) res[0][84]=a1;
  if (res[0]!=0) res[0][85]=a0;
  if (res[0]!=0) res[0][86]=a0;
  if (res[0]!=0) res[0][87]=a0;
  if (res[0]!=0) res[0][88]=a0;
  if (res[0]!=0) res[0][89]=a0;
  if (res[0]!=0) res[0][90]=a0;
  if (res[0]!=0) res[0][91]=a0;
  if (res[0]!=0) res[0][92]=a0;
  if (res[0]!=0) res[0][93]=a0;
  if (res[0]!=0) res[0][94]=a0;
  if (res[0]!=0) res[0][95]=a0;
  if (res[0]!=0) res[0][96]=a1;
  if (res[0]!=0) res[0][97]=a0;
  if (res[0]!=0) res[0][98]=a0;
  if (res[0]!=0) res[0][99]=a0;
  if (res[0]!=0) res[0][100]=a0;
  if (res[0]!=0) res[0][101]=a0;
  if (res[0]!=0) res[0][102]=a0;
  if (res[0]!=0) res[0][103]=a0;
  if (res[0]!=0) res[0][104]=a0;
  if (res[0]!=0) res[0][105]=a0;
  if (res[0]!=0) res[0][106]=a0;
  if (res[0]!=0) res[0][107]=a0;
  if (res[0]!=0) res[0][108]=a1;
  if (res[0]!=0) res[0][109]=a0;
  if (res[0]!=0) res[0][110]=a0;
  if (res[0]!=0) res[0][111]=a0;
  if (res[0]!=0) res[0][112]=a0;
  if (res[0]!=0) res[0][113]=a0;
  if (res[0]!=0) res[0][114]=a0;
  if (res[0]!=0) res[0][115]=a0;
  if (res[0]!=0) res[0][116]=a0;
  if (res[0]!=0) res[0][117]=a0;
  if (res[0]!=0) res[0][118]=a0;
  if (res[0]!=0) res[0][119]=a0;
  if (res[0]!=0) res[0][120]=a1;
  if (res[1]!=0) res[1][0]=a0;
  if (res[1]!=0) res[1][1]=a0;
  if (res[1]!=0) res[1][2]=a0;
  if (res[1]!=0) res[1][3]=a0;
  if (res[1]!=0) res[1][4]=a0;
  if (res[1]!=0) res[1][5]=a0;
  if (res[1]!=0) res[1][6]=a0;
  a1=-25.;
  if (res[1]!=0) res[1][7]=a1;
  if (res[1]!=0) res[1][8]=a0;
  if (res[1]!=0) res[1][9]=a0;
  if (res[1]!=0) res[1][10]=a0;
  if (res[1]!=0) res[1][11]=a0;
  if (res[1]!=0) res[1][12]=a0;
  if (res[1]!=0) res[1][13]=a0;
  if (res[1]!=0) res[1][14]=a0;
  if (res[1]!=0) res[1][15]=a0;
  if (res[1]!=0) res[1][16]=a0;
  if (res[1]!=0) res[1][17]=a0;
  if (res[1]!=0) res[1][18]=a0;
  if (res[1]!=0) res[1][19]=a1;
  if (res[1]!=0) res[1][20]=a0;
  if (res[1]!=0) res[1][21]=a0;
  if (res[1]!=0) res[1][22]=a0;
  if (res[1]!=0) res[1][23]=a0;
  if (res[1]!=0) res[1][24]=a0;
  if (res[1]!=0) res[1][25]=a0;
  if (res[1]!=0) res[1][26]=a0;
  if (res[1]!=0) res[1][27]=a0;
  if (res[1]!=0) res[1][28]=a0;
  if (res[1]!=0) res[1][29]=a0;
  if (res[1]!=0) res[1][30]=a0;
  if (res[1]!=0) res[1][31]=a1;
  if (res[1]!=0) res[1][32]=a0;
  if (res[1]!=0) res[1][33]=a0;
  if (res[1]!=0) res[1][34]=a0;
  if (res[1]!=0) res[1][35]=a0;
  if (res[1]!=0) res[1][36]=a0;
  if (res[1]!=0) res[1][37]=a0;
  if (res[1]!=0) res[1][38]=a0;
  if (res[1]!=0) res[1][39]=a0;
  if (res[1]!=0) res[1][40]=a0;
  if (res[1]!=0) res[1][41]=a0;
  if (res[1]!=0) res[1][42]=a0;
  if (res[1]!=0) res[1][43]=a1;
  a1=1.;
  if (res[2]!=0) res[2][0]=a1;
  if (res[2]!=0) res[2][1]=a0;
  if (res[2]!=0) res[2][2]=a0;
  if (res[2]!=0) res[2][3]=a0;
  if (res[2]!=0) res[2][4]=a0;
  if (res[2]!=0) res[2][5]=a0;
  if (res[2]!=0) res[2][6]=a0;
  if (res[2]!=0) res[2][7]=a0;
  if (res[2]!=0) res[2][8]=a0;
  if (res[2]!=0) res[2][9]=a0;
  if (res[2]!=0) res[2][10]=a0;
  if (res[2]!=0) res[2][11]=a0;
  if (res[2]!=0) res[2][12]=a1;
  if (res[2]!=0) res[2][13]=a0;
  if (res[2]!=0) res[2][14]=a0;
  if (res[2]!=0) res[2][15]=a0;
  if (res[2]!=0) res[2][16]=a0;
  if (res[2]!=0) res[2][17]=a0;
  if (res[2]!=0) res[2][18]=a0;
  if (res[2]!=0) res[2][19]=a0;
  if (res[2]!=0) res[2][20]=a0;
  if (res[2]!=0) res[2][21]=a0;
  if (res[2]!=0) res[2][22]=a0;
  if (res[2]!=0) res[2][23]=a0;
  if (res[2]!=0) res[2][24]=a1;
  if (res[2]!=0) res[2][25]=a0;
  if (res[2]!=0) res[2][26]=a0;
  if (res[2]!=0) res[2][27]=a0;
  if (res[2]!=0) res[2][28]=a0;
  if (res[2]!=0) res[2][29]=a0;
  if (res[2]!=0) res[2][30]=a0;
  if (res[2]!=0) res[2][31]=a0;
  if (res[2]!=0) res[2][32]=a0;
  if (res[2]!=0) res[2][33]=a0;
  if (res[2]!=0) res[2][34]=a0;
  if (res[2]!=0) res[2][35]=a0;
  if (res[2]!=0) res[2][36]=a1;
  if (res[2]!=0) res[2][37]=a0;
  if (res[2]!=0) res[2][38]=a0;
  if (res[2]!=0) res[2][39]=a0;
  if (res[2]!=0) res[2][40]=a0;
  if (res[2]!=0) res[2][41]=a0;
  if (res[2]!=0) res[2][42]=a0;
  if (res[2]!=0) res[2][43]=a0;
  if (res[2]!=0) res[2][44]=a0;
  if (res[2]!=0) res[2][45]=a0;
  if (res[2]!=0) res[2][46]=a0;
  if (res[2]!=0) res[2][47]=a0;
  if (res[2]!=0) res[2][48]=a1;
  if (res[2]!=0) res[2][49]=a0;
  if (res[2]!=0) res[2][50]=a0;
  if (res[2]!=0) res[2][51]=a0;
  if (res[2]!=0) res[2][52]=a0;
  if (res[2]!=0) res[2][53]=a0;
  if (res[2]!=0) res[2][54]=a0;
  if (res[2]!=0) res[2][55]=a0;
  if (res[2]!=0) res[2][56]=a0;
  if (res[2]!=0) res[2][57]=a0;
  if (res[2]!=0) res[2][58]=a0;
  if (res[2]!=0) res[2][59]=a0;
  if (res[2]!=0) res[2][60]=a1;
  if (res[2]!=0) res[2][61]=a0;
  if (res[2]!=0) res[2][62]=a0;
  if (res[2]!=0) res[2][63]=a0;
  if (res[2]!=0) res[2][64]=a0;
  if (res[2]!=0) res[2][65]=a0;
  if (res[2]!=0) res[2][66]=a0;
  if (res[2]!=0) res[2][67]=a0;
  if (res[2]!=0) res[2][68]=a0;
  if (res[2]!=0) res[2][69]=a0;
  if (res[2]!=0) res[2][70]=a0;
  if (res[2]!=0) res[2][71]=a0;
  if (res[2]!=0) res[2][72]=a1;
  if (res[2]!=0) res[2][73]=a0;
  if (res[2]!=0) res[2][74]=a0;
  if (res[2]!=0) res[2][75]=a0;
  if (res[2]!=0) res[2][76]=a0;
  a2=-1.;
  if (res[3]!=0) res[3][0]=a2;
  if (res[3]!=0) res[3][1]=a0;
  if (res[3]!=0) res[3][2]=a0;
  if (res[3]!=0) res[3][3]=a0;
  if (res[3]!=0) res[3][4]=a0;
  if (res[3]!=0) res[3][5]=a0;
  if (res[3]!=0) res[3][6]=a0;
  if (res[3]!=0) res[3][7]=a0;
  if (res[3]!=0) res[3][8]=a0;
  if (res[3]!=0) res[3][9]=a0;
  if (res[3]!=0) res[3][10]=a0;
  if (res[3]!=0) res[3][11]=a0;
  if (res[3]!=0) res[3][12]=a2;
  if (res[3]!=0) res[3][13]=a0;
  if (res[3]!=0) res[3][14]=a0;
  if (res[3]!=0) res[3][15]=a0;
  if (res[3]!=0) res[3][16]=a0;
  if (res[3]!=0) res[3][17]=a0;
  if (res[3]!=0) res[3][18]=a0;
  if (res[3]!=0) res[3][19]=a0;
  if (res[3]!=0) res[3][20]=a0;
  if (res[3]!=0) res[3][21]=a0;
  if (res[3]!=0) res[3][22]=a0;
  if (res[3]!=0) res[3][23]=a0;
  if (res[3]!=0) res[3][24]=a2;
  if (res[3]!=0) res[3][25]=a0;
  if (res[3]!=0) res[3][26]=a0;
  if (res[3]!=0) res[3][27]=a0;
  if (res[3]!=0) res[3][28]=a0;
  if (res[3]!=0) res[3][29]=a0;
  if (res[3]!=0) res[3][30]=a0;
  if (res[3]!=0) res[3][31]=a0;
  if (res[3]!=0) res[3][32]=a0;
  if (res[3]!=0) res[3][33]=a0;
  if (res[3]!=0) res[3][34]=a0;
  if (res[3]!=0) res[3][35]=a0;
  if (res[3]!=0) res[3][36]=a2;
  if (res[3]!=0) res[3][37]=a0;
  if (res[3]!=0) res[3][38]=a0;
  if (res[3]!=0) res[3][39]=a0;
  if (res[3]!=0) res[3][40]=a0;
  if (res[3]!=0) res[3][41]=a0;
  if (res[3]!=0) res[3][42]=a0;
  if (res[3]!=0) res[3][43]=a0;
  if (res[3]!=0) res[3][44]=a0;
  if (res[3]!=0) res[3][45]=a0;
  if (res[3]!=0) res[3][46]=a0;
  if (res[3]!=0) res[3][47]=a0;
  if (res[3]!=0) res[3][48]=a2;
  if (res[3]!=0) res[3][49]=a0;
  if (res[3]!=0) res[3][50]=a0;
  if (res[3]!=0) res[3][51]=a0;
  if (res[3]!=0) res[3][52]=a0;
  if (res[3]!=0) res[3][53]=a0;
  if (res[3]!=0) res[3][54]=a0;
  if (res[3]!=0) res[3][55]=a0;
  if (res[3]!=0) res[3][56]=a0;
  if (res[3]!=0) res[3][57]=a0;
  if (res[3]!=0) res[3][58]=a0;
  if (res[3]!=0) res[3][59]=a0;
  if (res[3]!=0) res[3][60]=a2;
  if (res[3]!=0) res[3][61]=a0;
  if (res[3]!=0) res[3][62]=a0;
  if (res[3]!=0) res[3][63]=a0;
  if (res[3]!=0) res[3][64]=a0;
  if (res[3]!=0) res[3][65]=a0;
  if (res[3]!=0) res[3][66]=a0;
  if (res[3]!=0) res[3][67]=a0;
  if (res[3]!=0) res[3][68]=a0;
  if (res[3]!=0) res[3][69]=a0;
  if (res[3]!=0) res[3][70]=a0;
  if (res[3]!=0) res[3][71]=a0;
  if (res[3]!=0) res[3][72]=a2;
  if (res[3]!=0) res[3][73]=a0;
  if (res[3]!=0) res[3][74]=a0;
  if (res[3]!=0) res[3][75]=a0;
  if (res[3]!=0) res[3][76]=a0;
  if (res[3]!=0) res[3][77]=a0;
  if (res[3]!=0) res[3][78]=a0;
  if (res[3]!=0) res[3][79]=a0;
  if (res[3]!=0) res[3][80]=a0;
  if (res[3]!=0) res[3][81]=a0;
  if (res[3]!=0) res[3][82]=a0;
  if (res[3]!=0) res[3][83]=a0;
  if (res[3]!=0) res[3][84]=a2;
  if (res[3]!=0) res[3][85]=a0;
  if (res[3]!=0) res[3][86]=a0;
  if (res[3]!=0) res[3][87]=a0;
  if (res[3]!=0) res[3][88]=a0;
  if (res[3]!=0) res[3][89]=a0;
  if (res[3]!=0) res[3][90]=a0;
  if (res[3]!=0) res[3][91]=a0;
  if (res[3]!=0) res[3][92]=a0;
  if (res[3]!=0) res[3][93]=a0;
  if (res[3]!=0) res[3][94]=a0;
  if (res[3]!=0) res[3][95]=a0;
  if (res[3]!=0) res[3][96]=a2;
  if (res[3]!=0) res[3][97]=a0;
  if (res[3]!=0) res[3][98]=a0;
  if (res[3]!=0) res[3][99]=a0;
  if (res[3]!=0) res[3][100]=a0;
  if (res[3]!=0) res[3][101]=a0;
  if (res[3]!=0) res[3][102]=a0;
  if (res[3]!=0) res[3][103]=a0;
  if (res[3]!=0) res[3][104]=a0;
  if (res[3]!=0) res[3][105]=a0;
  if (res[3]!=0) res[3][106]=a0;
  if (res[3]!=0) res[3][107]=a0;
  if (res[3]!=0) res[3][108]=a2;
  if (res[3]!=0) res[3][109]=a0;
  if (res[3]!=0) res[3][110]=a0;
  if (res[3]!=0) res[3][111]=a0;
  if (res[3]!=0) res[3][112]=a0;
  if (res[3]!=0) res[3][113]=a0;
  if (res[3]!=0) res[3][114]=a0;
  if (res[3]!=0) res[3][115]=a0;
  if (res[3]!=0) res[3][116]=a0;
  if (res[3]!=0) res[3][117]=a0;
  if (res[3]!=0) res[3][118]=a0;
  if (res[3]!=0) res[3][119]=a0;
  if (res[3]!=0) res[3][120]=a2;
  if (res[4]!=0) res[4][0]=a1;
  if (res[4]!=0) res[4][1]=a0;
  if (res[4]!=0) res[4][2]=a0;
  if (res[4]!=0) res[4][3]=a0;
  if (res[4]!=0) res[4][4]=a0;
  if (res[4]!=0) res[4][5]=a0;
  if (res[4]!=0) res[4][6]=a0;
  if (res[4]!=0) res[4][7]=a0;
  if (res[4]!=0) res[4][8]=a0;
  if (res[4]!=0) res[4][9]=a0;
  if (res[4]!=0) res[4][10]=a0;
  if (res[4]!=0) res[4][11]=a0;
  if (res[4]!=0) res[4][12]=a1;
  if (res[4]!=0) res[4][13]=a0;
  if (res[4]!=0) res[4][14]=a0;
  if (res[4]!=0) res[4][15]=a0;
  if (res[4]!=0) res[4][16]=a0;
  if (res[4]!=0) res[4][17]=a0;
  if (res[4]!=0) res[4][18]=a0;
  if (res[4]!=0) res[4][19]=a0;
  if (res[4]!=0) res[4][20]=a0;
  if (res[4]!=0) res[4][21]=a0;
  if (res[4]!=0) res[4][22]=a0;
  if (res[4]!=0) res[4][23]=a0;
  if (res[4]!=0) res[4][24]=a1;
  if (res[4]!=0) res[4][25]=a0;
  if (res[4]!=0) res[4][26]=a0;
  if (res[4]!=0) res[4][27]=a0;
  if (res[4]!=0) res[4][28]=a0;
  if (res[4]!=0) res[4][29]=a0;
  if (res[4]!=0) res[4][30]=a0;
  if (res[4]!=0) res[4][31]=a0;
  if (res[4]!=0) res[4][32]=a0;
  if (res[4]!=0) res[4][33]=a0;
  if (res[4]!=0) res[4][34]=a0;
  if (res[4]!=0) res[4][35]=a0;
  if (res[4]!=0) res[4][36]=a1;
  if (res[4]!=0) res[4][37]=a0;
  if (res[4]!=0) res[4][38]=a0;
  if (res[4]!=0) res[4][39]=a0;
  if (res[4]!=0) res[4][40]=a0;
  if (res[4]!=0) res[4][41]=a0;
  if (res[4]!=0) res[4][42]=a0;
  if (res[4]!=0) res[4][43]=a0;
  if (res[4]!=0) res[4][44]=a0;
  if (res[4]!=0) res[4][45]=a0;
  if (res[4]!=0) res[4][46]=a0;
  if (res[4]!=0) res[4][47]=a0;
  if (res[4]!=0) res[4][48]=a1;
  if (res[4]!=0) res[4][49]=a0;
  if (res[4]!=0) res[4][50]=a0;
  if (res[4]!=0) res[4][51]=a0;
  if (res[4]!=0) res[4][52]=a0;
  if (res[4]!=0) res[4][53]=a0;
  if (res[4]!=0) res[4][54]=a0;
  if (res[4]!=0) res[4][55]=a0;
  if (res[4]!=0) res[4][56]=a0;
  if (res[4]!=0) res[4][57]=a0;
  if (res[4]!=0) res[4][58]=a0;
  if (res[4]!=0) res[4][59]=a0;
  if (res[4]!=0) res[4][60]=a1;
  if (res[4]!=0) res[4][61]=a0;
  if (res[4]!=0) res[4][62]=a0;
  if (res[4]!=0) res[4][63]=a0;
  if (res[4]!=0) res[4][64]=a0;
  if (res[4]!=0) res[4][65]=a0;
  if (res[4]!=0) res[4][66]=a0;
  if (res[4]!=0) res[4][67]=a0;
  if (res[4]!=0) res[4][68]=a0;
  if (res[4]!=0) res[4][69]=a0;
  if (res[4]!=0) res[4][70]=a0;
  if (res[4]!=0) res[4][71]=a0;
  if (res[4]!=0) res[4][72]=a1;
  if (res[4]!=0) res[4][73]=a0;
  if (res[4]!=0) res[4][74]=a0;
  if (res[4]!=0) res[4][75]=a0;
  if (res[4]!=0) res[4][76]=a0;
  if (res[4]!=0) res[4][77]=a0;
  if (res[4]!=0) res[4][78]=a0;
  if (res[4]!=0) res[4][79]=a0;
  if (res[4]!=0) res[4][80]=a0;
  if (res[4]!=0) res[4][81]=a0;
  if (res[4]!=0) res[4][82]=a0;
  if (res[4]!=0) res[4][83]=a0;
  if (res[4]!=0) res[4][84]=a1;
  if (res[4]!=0) res[4][85]=a0;
  if (res[4]!=0) res[4][86]=a0;
  if (res[4]!=0) res[4][87]=a0;
  if (res[4]!=0) res[4][88]=a0;
  if (res[4]!=0) res[4][89]=a0;
  if (res[4]!=0) res[4][90]=a0;
  if (res[4]!=0) res[4][91]=a0;
  if (res[4]!=0) res[4][92]=a0;
  if (res[4]!=0) res[4][93]=a0;
  if (res[4]!=0) res[4][94]=a0;
  if (res[4]!=0) res[4][95]=a0;
  if (res[4]!=0) res[4][96]=a1;
  if (res[4]!=0) res[4][97]=a0;
  if (res[4]!=0) res[4][98]=a0;
  if (res[4]!=0) res[4][99]=a0;
  if (res[4]!=0) res[4][100]=a0;
  if (res[4]!=0) res[4][101]=a0;
  if (res[4]!=0) res[4][102]=a0;
  if (res[4]!=0) res[4][103]=a0;
  if (res[4]!=0) res[4][104]=a0;
  if (res[4]!=0) res[4][105]=a0;
  if (res[4]!=0) res[4][106]=a0;
  if (res[4]!=0) res[4][107]=a0;
  if (res[4]!=0) res[4][108]=a1;
  if (res[4]!=0) res[4][109]=a0;
  if (res[4]!=0) res[4][110]=a0;
  if (res[4]!=0) res[4][111]=a0;
  if (res[4]!=0) res[4][112]=a0;
  if (res[4]!=0) res[4][113]=a0;
  if (res[4]!=0) res[4][114]=a0;
  if (res[4]!=0) res[4][115]=a0;
  if (res[4]!=0) res[4][116]=a0;
  if (res[4]!=0) res[4][117]=a0;
  if (res[4]!=0) res[4][118]=a0;
  if (res[4]!=0) res[4][119]=a0;
  if (res[4]!=0) res[4][120]=a1;
  if (res[5]!=0) res[5][0]=a0;
  if (res[5]!=0) res[5][1]=a0;
  if (res[5]!=0) res[5][2]=a0;
  if (res[5]!=0) res[5][3]=a0;
  if (res[5]!=0) res[5][4]=a0;
  if (res[5]!=0) res[5][5]=a0;
  if (res[5]!=0) res[5][6]=a0;
  if (res[5]!=0) res[5][7]=a0;
  if (res[5]!=0) res[5][8]=a0;
  if (res[5]!=0) res[5][9]=a0;
  if (res[5]!=0) res[5][10]=a0;
  if (res[5]!=0) res[5][11]=a0;
  if (res[5]!=0) res[5][12]=a0;
  if (res[5]!=0) res[5][13]=a0;
  if (res[5]!=0) res[5][14]=a0;
  if (res[5]!=0) res[5][15]=a0;
  if (res[5]!=0) res[5][16]=a0;
  if (res[5]!=0) res[5][17]=a0;
  if (res[5]!=0) res[5][18]=a0;
  if (res[5]!=0) res[5][19]=a0;
  if (res[5]!=0) res[5][20]=a0;
  if (res[5]!=0) res[5][21]=a0;
  if (res[5]!=0) res[5][22]=a0;
  if (res[5]!=0) res[5][23]=a0;
  if (res[5]!=0) res[5][24]=a0;
  if (res[5]!=0) res[5][25]=a0;
  if (res[5]!=0) res[5][26]=a0;
  if (res[5]!=0) res[5][27]=a0;
  if (res[5]!=0) res[5][28]=a0;
  if (res[5]!=0) res[5][29]=a0;
  if (res[5]!=0) res[5][30]=a0;
  if (res[5]!=0) res[5][31]=a0;
  if (res[5]!=0) res[5][32]=a0;
  if (res[5]!=0) res[5][33]=a0;
  if (res[5]!=0) res[5][34]=a0;
  if (res[5]!=0) res[5][35]=a0;
  if (res[5]!=0) res[5][36]=a0;
  if (res[5]!=0) res[5][37]=a0;
  if (res[5]!=0) res[5][38]=a0;
  if (res[5]!=0) res[5][39]=a0;
  if (res[5]!=0) res[5][40]=a0;
  if (res[5]!=0) res[5][41]=a0;
  if (res[5]!=0) res[5][42]=a0;
  if (res[5]!=0) res[5][43]=a0;
  if (res[5]!=0) res[5][44]=a0;
  if (res[5]!=0) res[5][45]=a0;
  if (res[5]!=0) res[5][46]=a0;
  if (res[5]!=0) res[5][47]=a0;
  if (res[5]!=0) res[5][48]=a0;
  if (res[5]!=0) res[5][49]=a0;
  if (res[5]!=0) res[5][50]=a0;
  if (res[5]!=0) res[5][51]=a0;
  if (res[5]!=0) res[5][52]=a0;
  if (res[5]!=0) res[5][53]=a0;
  if (res[5]!=0) res[5][54]=a0;
  if (res[5]!=0) res[5][55]=a0;
  if (res[5]!=0) res[5][56]=a0;
  if (res[5]!=0) res[5][57]=a0;
  if (res[5]!=0) res[5][58]=a0;
  if (res[5]!=0) res[5][59]=a0;
  if (res[5]!=0) res[5][60]=a0;
  if (res[5]!=0) res[5][61]=a0;
  if (res[5]!=0) res[5][62]=a0;
  if (res[5]!=0) res[5][63]=a0;
  if (res[5]!=0) res[5][64]=a0;
  if (res[5]!=0) res[5][65]=a0;
  if (res[5]!=0) res[5][66]=a0;
  if (res[5]!=0) res[5][67]=a0;
  if (res[5]!=0) res[5][68]=a0;
  if (res[5]!=0) res[5][69]=a0;
  if (res[5]!=0) res[5][70]=a0;
  if (res[5]!=0) res[5][71]=a0;
  if (res[5]!=0) res[5][72]=a0;
  if (res[5]!=0) res[5][73]=a0;
  if (res[5]!=0) res[5][74]=a0;
  if (res[5]!=0) res[5][75]=a0;
  if (res[5]!=0) res[5][76]=a0;
  if (res[5]!=0) res[5][77]=a0;
  if (res[5]!=0) res[5][78]=a0;
  if (res[5]!=0) res[5][79]=a0;
  if (res[5]!=0) res[5][80]=a0;
  if (res[5]!=0) res[5][81]=a0;
  if (res[5]!=0) res[5][82]=a0;
  if (res[5]!=0) res[5][83]=a0;
  if (res[5]!=0) res[5][84]=a0;
  if (res[5]!=0) res[5][85]=a0;
  if (res[5]!=0) res[5][86]=a0;
  if (res[5]!=0) res[5][87]=a0;
  if (res[5]!=0) res[5][88]=a0;
  if (res[5]!=0) res[5][89]=a0;
  if (res[5]!=0) res[5][90]=a0;
  if (res[5]!=0) res[5][91]=a0;
  if (res[5]!=0) res[5][92]=a0;
  if (res[5]!=0) res[5][93]=a0;
  if (res[5]!=0) res[5][94]=a0;
  if (res[5]!=0) res[5][95]=a0;
  if (res[5]!=0) res[5][96]=a0;
  if (res[5]!=0) res[5][97]=a0;
  if (res[5]!=0) res[5][98]=a0;
  if (res[5]!=0) res[5][99]=a0;
  if (res[5]!=0) res[5][100]=a0;
  if (res[5]!=0) res[5][101]=a0;
  if (res[5]!=0) res[5][102]=a0;
  if (res[5]!=0) res[5][103]=a0;
  if (res[5]!=0) res[5][104]=a0;
  if (res[5]!=0) res[5][105]=a0;
  if (res[5]!=0) res[5][106]=a0;
  if (res[5]!=0) res[5][107]=a0;
  if (res[5]!=0) res[5][108]=a0;
  if (res[5]!=0) res[5][109]=a0;
  if (res[5]!=0) res[5][110]=a0;
  if (res[5]!=0) res[5][111]=a0;
  if (res[5]!=0) res[5][112]=a0;
  if (res[5]!=0) res[5][113]=a0;
  if (res[5]!=0) res[5][114]=a0;
  if (res[5]!=0) res[5][115]=a0;
  if (res[5]!=0) res[5][116]=a0;
  if (res[5]!=0) res[5][117]=a0;
  if (res[5]!=0) res[5][118]=a0;
  if (res[5]!=0) res[5][119]=a0;
  if (res[5]!=0) res[5][120]=a0;
  if (res[8]!=0) res[8][0]=a0;
  if (res[8]!=0) res[8][1]=a0;
  if (res[8]!=0) res[8][2]=a0;
  if (res[8]!=0) res[8][3]=a0;
  if (res[8]!=0) res[8][4]=a0;
  if (res[8]!=0) res[8][5]=a0;
  if (res[8]!=0) res[8][6]=a0;
  if (res[8]!=0) res[8][7]=a0;
  if (res[8]!=0) res[8][8]=a0;
  if (res[8]!=0) res[8][9]=a0;
  if (res[8]!=0) res[8][10]=a0;
  if (res[8]!=0) res[8][11]=a0;
  if (res[8]!=0) res[8][12]=a0;
  if (res[8]!=0) res[8][13]=a0;
  if (res[8]!=0) res[8][14]=a0;
  if (res[8]!=0) res[8][15]=a0;
  if (res[8]!=0) res[8][16]=a0;
  if (res[8]!=0) res[8][17]=a0;
  if (res[8]!=0) res[8][18]=a2;
  if (res[8]!=0) res[8][19]=a0;
  if (res[8]!=0) res[8][20]=a0;
  if (res[8]!=0) res[8][21]=a0;
  if (res[8]!=0) res[8][22]=a0;
  if (res[8]!=0) res[8][23]=a0;
  if (res[8]!=0) res[8][24]=a0;
  if (res[8]!=0) res[8][25]=a2;
  if (res[8]!=0) res[8][26]=a0;
  if (res[8]!=0) res[8][27]=a0;
  if (res[8]!=0) res[8][28]=a0;
  if (res[8]!=0) res[8][29]=a0;
  if (res[8]!=0) res[8][30]=a0;
  if (res[8]!=0) res[8][31]=a0;
  if (res[8]!=0) res[8][32]=a2;
  if (res[8]!=0) res[8][33]=a0;
  if (res[8]!=0) res[8][34]=a0;
  if (res[8]!=0) res[8][35]=a0;
  if (res[9]!=0) res[9][0]=a0;
  if (res[9]!=0) res[9][1]=a0;
  if (res[9]!=0) res[9][2]=a0;
  if (res[9]!=0) res[9][3]=a0;
  if (res[9]!=0) res[9][4]=a0;
  if (res[9]!=0) res[9][5]=a0;
  if (res[9]!=0) res[9][6]=a0;
  if (res[9]!=0) res[9][7]=a0;
  if (res[9]!=0) res[9][8]=a0;
  if (res[9]!=0) res[9][9]=a0;
  if (res[9]!=0) res[9][10]=a0;
  if (res[10]!=0) res[10][0]=a2;
  if (res[10]!=0) res[10][1]=a0;
  if (res[10]!=0) res[10][2]=a0;
  if (res[10]!=0) res[10][3]=a0;
  if (res[10]!=0) res[10][4]=a0;
  if (res[10]!=0) res[10][5]=a0;
  if (res[10]!=0) res[10][6]=a0;
  if (res[10]!=0) res[10][7]=a2;
  if (res[10]!=0) res[10][8]=a0;
  if (res[10]!=0) res[10][9]=a0;
  if (res[10]!=0) res[10][10]=a0;
  if (res[10]!=0) res[10][11]=a0;
  if (res[10]!=0) res[10][12]=a0;
  if (res[10]!=0) res[10][13]=a0;
  if (res[10]!=0) res[10][14]=a2;
  if (res[10]!=0) res[10][15]=a0;
  if (res[10]!=0) res[10][16]=a0;
  if (res[10]!=0) res[10][17]=a0;
  if (res[10]!=0) res[10][18]=a0;
  if (res[10]!=0) res[10][19]=a0;
  if (res[10]!=0) res[10][20]=a0;
  if (res[10]!=0) res[10][21]=a2;
  if (res[10]!=0) res[10][22]=a0;
  if (res[10]!=0) res[10][23]=a0;
  if (res[10]!=0) res[10][24]=a0;
  if (res[10]!=0) res[10][25]=a0;
  if (res[10]!=0) res[10][26]=a0;
  if (res[10]!=0) res[10][27]=a0;
  if (res[10]!=0) res[10][28]=a2;
  if (res[10]!=0) res[10][29]=a0;
  if (res[10]!=0) res[10][30]=a0;
  if (res[10]!=0) res[10][31]=a0;
  if (res[10]!=0) res[10][32]=a0;
  if (res[10]!=0) res[10][33]=a0;
  if (res[10]!=0) res[10][34]=a0;
  if (res[10]!=0) res[10][35]=a2;
  if (res[11]!=0) res[11][0]=a0;
  if (res[11]!=0) res[11][1]=a0;
  if (res[11]!=0) res[11][2]=a0;
  if (res[11]!=0) res[11][3]=a0;
  if (res[11]!=0) res[11][4]=a0;
  if (res[11]!=0) res[11][5]=a0;
  if (res[11]!=0) res[11][6]=a0;
  if (res[11]!=0) res[11][7]=a0;
  if (res[11]!=0) res[11][8]=a0;
  if (res[11]!=0) res[11][9]=a0;
  if (res[11]!=0) res[11][10]=a0;
  if (res[11]!=0) res[11][11]=a0;
  if (res[11]!=0) res[11][12]=a0;
  if (res[11]!=0) res[11][13]=a0;
  if (res[11]!=0) res[11][14]=a0;
  if (res[11]!=0) res[11][15]=a0;
  if (res[11]!=0) res[11][16]=a0;
  if (res[11]!=0) res[11][17]=a0;
  if (res[11]!=0) res[11][18]=a0;
  if (res[11]!=0) res[11][19]=a0;
  if (res[11]!=0) res[11][20]=a0;
  if (res[11]!=0) res[11][21]=a0;
  if (res[11]!=0) res[11][22]=a0;
  if (res[11]!=0) res[11][23]=a0;
  if (res[12]!=0) res[12][0]=a1;
  if (res[13]!=0) res[13][0]=a0;
  a1=3.;
  if (res[14]!=0) res[14][0]=a1;
  a1=4.;
  if (res[14]!=0) res[14][1]=a1;
  a1=5.;
  if (res[14]!=0) res[14][2]=a1;
  a1=6.;
  if (res[14]!=0) res[14][3]=a1;
  a1=10.;
  if (res[14]!=0) res[14][4]=a1;
  a1=11.;
  if (res[14]!=0) res[14][5]=a1;
  a1=12.;
  if (res[14]!=0) res[14][6]=a1;
  a2=13.;
  if (res[14]!=0) res[14][7]=a2;
  a2=14.;
  if (res[14]!=0) res[14][8]=a2;
  a2=15.;
  if (res[14]!=0) res[14][9]=a2;
  a2=16.;
  if (res[14]!=0) res[14][10]=a2;
  if (res[14]!=0) res[14][11]=a1;
  if (res[14]!=0) res[14][12]=a2;
  if (res[14]!=0) res[14][13]=a2;
  if (res[14]!=0) res[14][14]=a2;
  if (res[14]!=0) res[14][15]=a2;
  if (res[14]!=0) res[14][16]=a2;
  if (res[16]!=0) res[16][0]=a0;
  if (res[16]!=0) res[16][1]=a0;
  if (res[16]!=0) res[16][2]=a0;
  if (res[16]!=0) res[16][3]=a0;
  if (res[16]!=0) res[16][4]=a0;
  if (res[16]!=0) res[16][5]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_gnsf_get_matrices_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_gnsf_get_matrices_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_gnsf_get_matrices_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_gnsf_get_matrices_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_gnsf_get_matrices_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_gnsf_get_matrices_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_gnsf_get_matrices_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_gnsf_get_matrices_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_gnsf_get_matrices_fun_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_gnsf_get_matrices_fun_n_out(void) { return 17;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_gnsf_get_matrices_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_gnsf_get_matrices_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_gnsf_get_matrices_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    case 5: return "o5";
    case 6: return "o6";
    case 7: return "o7";
    case 8: return "o8";
    case 9: return "o9";
    case 10: return "o10";
    case 11: return "o11";
    case 12: return "o12";
    case 13: return "o13";
    case 14: return "o14";
    case 15: return "o15";
    case 16: return "o16";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_gnsf_get_matrices_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_gnsf_get_matrices_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    case 1: return casadi_s2;
    case 2: return casadi_s3;
    case 3: return casadi_s1;
    case 4: return casadi_s1;
    case 5: return casadi_s1;
    case 6: return casadi_s4;
    case 7: return casadi_s5;
    case 8: return casadi_s6;
    case 9: return casadi_s7;
    case 10: return casadi_s6;
    case 11: return casadi_s8;
    case 12: return casadi_s0;
    case 13: return casadi_s0;
    case 14: return casadi_s9;
    case 15: return casadi_s10;
    case 16: return casadi_s11;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_gnsf_get_matrices_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 17;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_gnsf_get_matrices_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 17*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
