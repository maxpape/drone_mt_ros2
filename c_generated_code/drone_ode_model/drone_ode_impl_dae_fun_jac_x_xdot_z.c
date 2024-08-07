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
  #define CASADI_PREFIX(ID) drone_ode_impl_dae_fun_jac_x_xdot_z_ ## ID
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
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_sq CASADI_PREFIX(sq)

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

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[31] = {27, 1, 0, 27, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
static const casadi_int casadi_s4[93] = {17, 17, 0, 0, 0, 0, 6, 12, 18, 24, 25, 26, 27, 33, 39, 45, 52, 59, 66, 73, 4, 5, 6, 7, 8, 9, 3, 5, 6, 7, 8, 9, 3, 4, 6, 7, 8, 9, 3, 4, 5, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 11, 12, 3, 4, 5, 6, 10, 12, 3, 4, 5, 6, 10, 11, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 14, 7, 8, 9, 10, 11, 12, 15, 7, 8, 9, 10, 11, 12, 16};
static const casadi_int casadi_s5[37] = {17, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s6[3] = {17, 0, 0};

/* drone_ode_impl_dae_fun_jac_x_xdot_z:(i0[17],i1[17],i2[4],i3[],i4[],i5[27])->(o0[17],o1[17x17,73nz],o2[17x17,17nz],o3[17x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a5, a6, a7, a8, a9;
  a0=arg[1]? arg[1][0] : 0;
  a1=arg[0]? arg[0][7] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[1]? arg[1][1] : 0;
  a1=arg[0]? arg[0][8] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[1]? arg[1][2] : 0;
  a1=arg[0]? arg[0][9] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1]? arg[1][3] : 0;
  a1=arg[0]? arg[0][4] : 0;
  a2=arg[0]? arg[0][10] : 0;
  a3=(a1*a2);
  a4=arg[0]? arg[0][5] : 0;
  a5=arg[0]? arg[0][11] : 0;
  a6=(a4*a5);
  a3=(a3+a6);
  a6=arg[0]? arg[0][6] : 0;
  a7=arg[0]? arg[0][12] : 0;
  a8=(a6*a7);
  a3=(a3+a8);
  a8=2.;
  a3=(a3/a8);
  a0=(a0+a3);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1]? arg[1][4] : 0;
  a3=arg[0]? arg[0][3] : 0;
  a9=(a3*a2);
  a10=(a4*a7);
  a11=(a6*a5);
  a10=(a10-a11);
  a9=(a9+a10);
  a9=(a9/a8);
  a0=(a0-a9);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[1]? arg[1][5] : 0;
  a9=(a3*a5);
  a10=(a6*a2);
  a11=(a1*a7);
  a10=(a10-a11);
  a9=(a9+a10);
  a9=(a9/a8);
  a0=(a0-a9);
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[1]? arg[1][6] : 0;
  a9=(a3*a7);
  a10=(a1*a5);
  a11=(a4*a2);
  a10=(a10-a11);
  a9=(a9+a10);
  a9=(a9/a8);
  a0=(a0-a9);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a9=arg[0]? arg[0][13] : 0;
  a8=arg[0]? arg[0][14] : 0;
  a10=(a9+a8);
  a11=arg[0]? arg[0][15] : 0;
  a10=(a10+a11);
  a12=arg[0]? arg[0][16] : 0;
  a10=(a10+a12);
  a13=(a6*a10);
  a14=casadi_sq(a3);
  a15=casadi_sq(a1);
  a14=(a14+a15);
  a15=casadi_sq(a4);
  a14=(a14+a15);
  a15=casadi_sq(a6);
  a14=(a14+a15);
  a15=(a1/a14);
  a16=(a13*a15);
  a17=(a3/a14);
  a18=(a4*a10);
  a19=(a17*a18);
  a16=(a16+a19);
  a19=(a1*a10);
  a20=(a6/a14);
  a21=(a19*a20);
  a22=(a3*a10);
  a23=(a4/a14);
  a24=(a22*a23);
  a21=(a21+a24);
  a16=(a16+a21);
  a21=arg[5]? arg[5][0] : 0;
  a16=(a16/a21);
  a24=5.;
  a25=arg[5]? arg[5][21] : 0;
  a25=(a24*a25);
  a16=(a16+a25);
  a0=(a0-a16);
  if (res[0]!=0) res[0][7]=a0;
  a0=arg[1]? arg[1][8] : 0;
  a16=(a13*a23);
  a25=(a17*a19);
  a16=(a16-a25);
  a25=(a18*a20);
  a26=(a22*a15);
  a25=(a25-a26);
  a16=(a16+a25);
  a16=(a16/a21);
  a25=arg[5]? arg[5][22] : 0;
  a25=(a24*a25);
  a16=(a16+a25);
  a0=(a0-a16);
  if (res[0]!=0) res[0][8]=a0;
  a0=arg[1]? arg[1][9] : 0;
  a16=(a13*a20);
  a25=(a17*a22);
  a16=(a16+a25);
  a25=(a18*a23);
  a26=(a19*a15);
  a25=(a25+a26);
  a16=(a16-a25);
  a16=(a16/a21);
  a25=arg[5]? arg[5][1] : 0;
  a16=(a16+a25);
  a25=arg[5]? arg[5][23] : 0;
  a25=(a24*a25);
  a16=(a16+a25);
  a0=(a0-a16);
  if (res[0]!=0) res[0][9]=a0;
  a0=arg[1]? arg[1][10] : 0;
  a16=arg[5]? arg[5][6] : 0;
  a25=(a16*a8);
  a26=arg[5]? arg[5][5] : 0;
  a27=(a26*a9);
  a25=(a25-a27);
  a27=arg[5]? arg[5][7] : 0;
  a28=(a27*a11);
  a25=(a25+a28);
  a28=arg[5]? arg[5][8] : 0;
  a29=(a28*a12);
  a25=(a25-a29);
  a29=arg[5]? arg[5][4] : 0;
  a30=(a29*a7);
  a31=(a5*a30);
  a32=arg[5]? arg[5][3] : 0;
  a33=(a32*a5);
  a34=(a7*a33);
  a31=(a31-a34);
  a25=(a25-a31);
  a31=arg[5]? arg[5][2] : 0;
  a25=(a25/a31);
  a34=arg[5]? arg[5][24] : 0;
  a34=(a24*a34);
  a25=(a25+a34);
  a0=(a0-a25);
  if (res[0]!=0) res[0][10]=a0;
  a0=arg[1]? arg[1][11] : 0;
  a25=arg[5]? arg[5][10] : 0;
  a34=(a25*a8);
  a35=arg[5]? arg[5][9] : 0;
  a36=(a35*a9);
  a34=(a34-a36);
  a36=arg[5]? arg[5][11] : 0;
  a37=(a36*a11);
  a34=(a34-a37);
  a37=arg[5]? arg[5][12] : 0;
  a38=(a37*a12);
  a34=(a34+a38);
  a38=(a31*a2);
  a39=(a7*a38);
  a40=(a2*a30);
  a39=(a39-a40);
  a34=(a34-a39);
  a34=(a34/a32);
  a39=arg[5]? arg[5][25] : 0;
  a39=(a24*a39);
  a34=(a34+a39);
  a0=(a0-a34);
  if (res[0]!=0) res[0][11]=a0;
  a0=arg[1]? arg[1][12] : 0;
  a34=arg[5]? arg[5][13] : 0;
  a39=(a34*a8);
  a40=(a34*a9);
  a39=(a39-a40);
  a40=(a34*a11);
  a39=(a39-a40);
  a40=(a34*a12);
  a39=(a39+a40);
  a40=(a2*a33);
  a41=(a5*a38);
  a40=(a40-a41);
  a39=(a39-a40);
  a39=(a39/a29);
  a40=arg[5]? arg[5][26] : 0;
  a24=(a24*a40);
  a39=(a39+a24);
  a0=(a0-a39);
  if (res[0]!=0) res[0][12]=a0;
  a0=arg[1]? arg[1][13] : 0;
  a39=25.;
  a24=arg[2]? arg[2][0] : 0;
  a24=(a24-a9);
  a24=(a39*a24);
  a0=(a0-a24);
  if (res[0]!=0) res[0][13]=a0;
  a0=arg[1]? arg[1][14] : 0;
  a24=arg[2]? arg[2][1] : 0;
  a24=(a24-a8);
  a24=(a39*a24);
  a0=(a0-a24);
  if (res[0]!=0) res[0][14]=a0;
  a0=arg[1]? arg[1][15] : 0;
  a24=arg[2]? arg[2][2] : 0;
  a24=(a24-a11);
  a24=(a39*a24);
  a0=(a0-a24);
  if (res[0]!=0) res[0][15]=a0;
  a0=arg[1]? arg[1][16] : 0;
  a24=arg[2]? arg[2][3] : 0;
  a24=(a24-a12);
  a24=(a39*a24);
  a0=(a0-a24);
  if (res[0]!=0) res[0][16]=a0;
  a0=5.0000000000000000e-01;
  a24=(a0*a2);
  a24=(-a24);
  if (res[1]!=0) res[1][0]=a24;
  a24=(a0*a5);
  a24=(-a24);
  if (res[1]!=0) res[1][1]=a24;
  a24=(a0*a7);
  a24=(-a24);
  if (res[1]!=0) res[1][2]=a24;
  a24=(1./a14);
  a12=(a17/a14);
  a11=(a3+a3);
  a8=(a12*a11);
  a24=(a24-a8);
  a8=(a18*a24);
  a9=(a15/a14);
  a40=(a9*a11);
  a41=(a13*a40);
  a8=(a8-a41);
  a41=(a23*a10);
  a42=(a23/a14);
  a43=(a42*a11);
  a44=(a22*a43);
  a41=(a41-a44);
  a44=(a20/a14);
  a11=(a44*a11);
  a45=(a19*a11);
  a41=(a41-a45);
  a8=(a8+a41);
  a8=(a8/a21);
  a8=(-a8);
  if (res[1]!=0) res[1][3]=a8;
  a8=(a13*a43);
  a41=(a19*a24);
  a8=(a8+a41);
  a41=(a18*a11);
  a45=(a15*a10);
  a46=(a22*a40);
  a45=(a45-a46);
  a41=(a41+a45);
  a8=(a8+a41);
  a8=(a8/a21);
  if (res[1]!=0) res[1][4]=a8;
  a24=(a22*a24);
  a8=(a17*a10);
  a24=(a24+a8);
  a11=(a13*a11);
  a24=(a24-a11);
  a43=(a18*a43);
  a40=(a19*a40);
  a43=(a43+a40);
  a24=(a24+a43);
  a24=(a24/a21);
  a24=(-a24);
  if (res[1]!=0) res[1][5]=a24;
  a24=(a0*a2);
  if (res[1]!=0) res[1][6]=a24;
  a24=(a0*a7);
  if (res[1]!=0) res[1][7]=a24;
  a24=(a0*a5);
  a24=(-a24);
  if (res[1]!=0) res[1][8]=a24;
  a24=(1./a14);
  a43=(a1+a1);
  a40=(a9*a43);
  a24=(a24-a40);
  a40=(a13*a24);
  a11=(a12*a43);
  a8=(a18*a11);
  a40=(a40-a8);
  a8=(a20*a10);
  a41=(a44*a43);
  a45=(a19*a41);
  a8=(a8-a45);
  a43=(a42*a43);
  a45=(a22*a43);
  a8=(a8-a45);
  a40=(a40+a8);
  a40=(a40/a21);
  a40=(-a40);
  if (res[1]!=0) res[1][9]=a40;
  a40=(a13*a43);
  a8=(a17*a10);
  a45=(a19*a11);
  a8=(a8-a45);
  a40=(a40+a8);
  a8=(a18*a41);
  a45=(a22*a24);
  a8=(a8+a45);
  a40=(a40+a8);
  a40=(a40/a21);
  if (res[1]!=0) res[1][10]=a40;
  a41=(a13*a41);
  a11=(a22*a11);
  a41=(a41+a11);
  a11=(a15*a10);
  a24=(a19*a24);
  a11=(a11+a24);
  a43=(a18*a43);
  a11=(a11-a43);
  a41=(a41+a11);
  a41=(a41/a21);
  if (res[1]!=0) res[1][11]=a41;
  a41=(a0*a5);
  if (res[1]!=0) res[1][12]=a41;
  a41=(a0*a7);
  a41=(-a41);
  if (res[1]!=0) res[1][13]=a41;
  a41=(a0*a2);
  if (res[1]!=0) res[1][14]=a41;
  a41=(a17*a10);
  a11=(a4+a4);
  a43=(a12*a11);
  a24=(a18*a43);
  a41=(a41-a24);
  a24=(a9*a11);
  a40=(a13*a24);
  a41=(a41-a40);
  a40=(1./a14);
  a8=(a42*a11);
  a40=(a40-a8);
  a8=(a22*a40);
  a11=(a44*a11);
  a45=(a19*a11);
  a8=(a8-a45);
  a41=(a41+a8);
  a41=(a41/a21);
  a41=(-a41);
  if (res[1]!=0) res[1][15]=a41;
  a41=(a13*a40);
  a8=(a19*a43);
  a41=(a41+a8);
  a8=(a20*a10);
  a45=(a18*a11);
  a8=(a8-a45);
  a45=(a22*a24);
  a8=(a8+a45);
  a41=(a41+a8);
  a41=(a41/a21);
  a41=(-a41);
  if (res[1]!=0) res[1][16]=a41;
  a11=(a13*a11);
  a43=(a22*a43);
  a11=(a11+a43);
  a43=(a23*a10);
  a40=(a18*a40);
  a43=(a43+a40);
  a24=(a19*a24);
  a43=(a43-a24);
  a11=(a11+a43);
  a11=(a11/a21);
  if (res[1]!=0) res[1][17]=a11;
  a11=(a0*a7);
  if (res[1]!=0) res[1][18]=a11;
  a11=(a0*a5);
  if (res[1]!=0) res[1][19]=a11;
  a11=(a0*a2);
  a11=(-a11);
  if (res[1]!=0) res[1][20]=a11;
  a11=(a15*a10);
  a43=(a6+a6);
  a9=(a9*a43);
  a24=(a13*a9);
  a11=(a11-a24);
  a12=(a12*a43);
  a24=(a18*a12);
  a11=(a11-a24);
  a14=(1./a14);
  a44=(a44*a43);
  a14=(a14-a44);
  a44=(a19*a14);
  a42=(a42*a43);
  a43=(a22*a42);
  a44=(a44-a43);
  a11=(a11+a44);
  a11=(a11/a21);
  a11=(-a11);
  if (res[1]!=0) res[1][21]=a11;
  a11=(a23*a10);
  a44=(a13*a42);
  a11=(a11-a44);
  a44=(a19*a12);
  a11=(a11+a44);
  a44=(a18*a14);
  a43=(a22*a9);
  a44=(a44+a43);
  a11=(a11+a44);
  a11=(a11/a21);
  a11=(-a11);
  if (res[1]!=0) res[1][22]=a11;
  a10=(a20*a10);
  a13=(a13*a14);
  a10=(a10+a13);
  a22=(a22*a12);
  a10=(a10-a22);
  a18=(a18*a42);
  a19=(a19*a9);
  a18=(a18+a19);
  a10=(a10+a18);
  a10=(a10/a21);
  a10=(-a10);
  if (res[1]!=0) res[1][23]=a10;
  a10=-1.;
  if (res[1]!=0) res[1][24]=a10;
  if (res[1]!=0) res[1][25]=a10;
  if (res[1]!=0) res[1][26]=a10;
  a10=(a0*a1);
  if (res[1]!=0) res[1][27]=a10;
  a10=(a0*a3);
  a10=(-a10);
  if (res[1]!=0) res[1][28]=a10;
  a10=(a0*a6);
  a10=(-a10);
  if (res[1]!=0) res[1][29]=a10;
  a10=(a0*a4);
  if (res[1]!=0) res[1][30]=a10;
  a10=(a7*a31);
  a10=(a10-a30);
  a10=(a10/a32);
  if (res[1]!=0) res[1][31]=a10;
  a10=(a5*a31);
  a10=(a33-a10);
  a10=(a10/a29);
  if (res[1]!=0) res[1][32]=a10;
  a10=(a0*a4);
  if (res[1]!=0) res[1][33]=a10;
  a10=(a0*a6);
  if (res[1]!=0) res[1][34]=a10;
  a10=(a0*a3);
  a10=(-a10);
  if (res[1]!=0) res[1][35]=a10;
  a10=(a0*a1);
  a10=(-a10);
  if (res[1]!=0) res[1][36]=a10;
  a7=(a7*a32);
  a30=(a30-a7);
  a30=(a30/a31);
  if (res[1]!=0) res[1][37]=a30;
  a30=(a2*a32);
  a30=(a30-a38);
  a30=(a30/a29);
  if (res[1]!=0) res[1][38]=a30;
  a30=(a0*a6);
  if (res[1]!=0) res[1][39]=a30;
  a30=(a0*a4);
  a30=(-a30);
  if (res[1]!=0) res[1][40]=a30;
  a30=(a0*a1);
  if (res[1]!=0) res[1][41]=a30;
  a0=(a0*a3);
  a0=(-a0);
  if (res[1]!=0) res[1][42]=a0;
  a5=(a5*a29);
  a5=(a5-a33);
  a5=(a5/a31);
  if (res[1]!=0) res[1][43]=a5;
  a2=(a2*a29);
  a38=(a38-a2);
  a38=(a38/a32);
  if (res[1]!=0) res[1][44]=a38;
  a38=(a15*a6);
  a2=(a17*a4);
  a38=(a38+a2);
  a2=(a20*a1);
  a5=(a23*a3);
  a2=(a2+a5);
  a38=(a38+a2);
  a38=(a38/a21);
  a38=(-a38);
  if (res[1]!=0) res[1][45]=a38;
  a38=(a23*a6);
  a2=(a17*a1);
  a38=(a38-a2);
  a2=(a20*a4);
  a5=(a15*a3);
  a2=(a2-a5);
  a38=(a38+a2);
  a38=(a38/a21);
  a38=(-a38);
  if (res[1]!=0) res[1][46]=a38;
  a38=(a20*a6);
  a2=(a17*a3);
  a38=(a38+a2);
  a2=(a23*a4);
  a5=(a15*a1);
  a2=(a2+a5);
  a38=(a38-a2);
  a38=(a38/a21);
  a38=(-a38);
  if (res[1]!=0) res[1][47]=a38;
  a26=(a26/a31);
  if (res[1]!=0) res[1][48]=a26;
  a35=(a35/a32);
  if (res[1]!=0) res[1][49]=a35;
  a35=(a34/a29);
  if (res[1]!=0) res[1][50]=a35;
  if (res[1]!=0) res[1][51]=a39;
  a35=(a15*a6);
  a26=(a17*a4);
  a35=(a35+a26);
  a26=(a20*a1);
  a38=(a23*a3);
  a26=(a26+a38);
  a35=(a35+a26);
  a35=(a35/a21);
  a35=(-a35);
  if (res[1]!=0) res[1][52]=a35;
  a35=(a23*a6);
  a26=(a17*a1);
  a35=(a35-a26);
  a26=(a20*a4);
  a38=(a15*a3);
  a26=(a26-a38);
  a35=(a35+a26);
  a35=(a35/a21);
  a35=(-a35);
  if (res[1]!=0) res[1][53]=a35;
  a35=(a20*a6);
  a26=(a17*a3);
  a35=(a35+a26);
  a26=(a23*a4);
  a38=(a15*a1);
  a26=(a26+a38);
  a35=(a35-a26);
  a35=(a35/a21);
  a35=(-a35);
  if (res[1]!=0) res[1][54]=a35;
  a16=(a16/a31);
  a16=(-a16);
  if (res[1]!=0) res[1][55]=a16;
  a25=(a25/a32);
  a25=(-a25);
  if (res[1]!=0) res[1][56]=a25;
  a25=(a34/a29);
  a25=(-a25);
  if (res[1]!=0) res[1][57]=a25;
  if (res[1]!=0) res[1][58]=a39;
  a25=(a15*a6);
  a16=(a17*a4);
  a25=(a25+a16);
  a16=(a20*a1);
  a35=(a23*a3);
  a16=(a16+a35);
  a25=(a25+a16);
  a25=(a25/a21);
  a25=(-a25);
  if (res[1]!=0) res[1][59]=a25;
  a25=(a23*a6);
  a16=(a17*a1);
  a25=(a25-a16);
  a16=(a20*a4);
  a35=(a15*a3);
  a16=(a16-a35);
  a25=(a25+a16);
  a25=(a25/a21);
  a25=(-a25);
  if (res[1]!=0) res[1][60]=a25;
  a25=(a20*a6);
  a16=(a17*a3);
  a25=(a25+a16);
  a16=(a23*a4);
  a35=(a15*a1);
  a16=(a16+a35);
  a25=(a25-a16);
  a25=(a25/a21);
  a25=(-a25);
  if (res[1]!=0) res[1][61]=a25;
  a27=(a27/a31);
  a27=(-a27);
  if (res[1]!=0) res[1][62]=a27;
  a36=(a36/a32);
  if (res[1]!=0) res[1][63]=a36;
  a36=(a34/a29);
  if (res[1]!=0) res[1][64]=a36;
  if (res[1]!=0) res[1][65]=a39;
  a36=(a15*a6);
  a27=(a17*a4);
  a36=(a36+a27);
  a27=(a20*a1);
  a25=(a23*a3);
  a27=(a27+a25);
  a36=(a36+a27);
  a36=(a36/a21);
  a36=(-a36);
  if (res[1]!=0) res[1][66]=a36;
  a36=(a23*a6);
  a27=(a17*a1);
  a36=(a36-a27);
  a27=(a20*a4);
  a25=(a15*a3);
  a27=(a27-a25);
  a36=(a36+a27);
  a36=(a36/a21);
  a36=(-a36);
  if (res[1]!=0) res[1][67]=a36;
  a20=(a20*a6);
  a17=(a17*a3);
  a20=(a20+a17);
  a23=(a23*a4);
  a15=(a15*a1);
  a23=(a23+a15);
  a20=(a20-a23);
  a20=(a20/a21);
  a20=(-a20);
  if (res[1]!=0) res[1][68]=a20;
  a28=(a28/a31);
  if (res[1]!=0) res[1][69]=a28;
  a37=(a37/a32);
  a37=(-a37);
  if (res[1]!=0) res[1][70]=a37;
  a34=(a34/a29);
  a34=(-a34);
  if (res[1]!=0) res[1][71]=a34;
  if (res[1]!=0) res[1][72]=a39;
  a39=1.;
  if (res[2]!=0) res[2][0]=a39;
  if (res[2]!=0) res[2][1]=a39;
  if (res[2]!=0) res[2][2]=a39;
  if (res[2]!=0) res[2][3]=a39;
  if (res[2]!=0) res[2][4]=a39;
  if (res[2]!=0) res[2][5]=a39;
  if (res[2]!=0) res[2][6]=a39;
  if (res[2]!=0) res[2][7]=a39;
  if (res[2]!=0) res[2][8]=a39;
  if (res[2]!=0) res[2][9]=a39;
  if (res[2]!=0) res[2][10]=a39;
  if (res[2]!=0) res[2][11]=a39;
  if (res[2]!=0) res[2][12]=a39;
  if (res[2]!=0) res[2][13]=a39;
  if (res[2]!=0) res[2][14]=a39;
  if (res[2]!=0) res[2][15]=a39;
  if (res[2]!=0) res[2][16]=a39;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_jac_x_xdot_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_jac_x_xdot_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_jac_x_xdot_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_jac_x_xdot_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_impl_dae_fun_jac_x_xdot_z_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_impl_dae_fun_jac_x_xdot_z_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_impl_dae_fun_jac_x_xdot_z_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_impl_dae_fun_jac_x_xdot_z_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    case 5: return "i5";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_impl_dae_fun_jac_x_xdot_z_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_impl_dae_fun_jac_x_xdot_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    case 5: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_impl_dae_fun_jac_x_xdot_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_z_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 4*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
