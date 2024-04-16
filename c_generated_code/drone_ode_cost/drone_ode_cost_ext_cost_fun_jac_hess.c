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
  #define CASADI_PREFIX(ID) drone_ode_cost_ext_cost_fun_jac_hess_ ## ID
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
#define casadi_fabs CASADI_PREFIX(fabs)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
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

casadi_real casadi_fabs(casadi_real x) {
/* Pre-c99 compatibility */
#if __STDC_VERSION__ < 199901L
  return x>0 ? x : -x;
#else
  return fabs(x);
#endif
}

static const casadi_int casadi_s0[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[31] = {27, 1, 0, 27, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s5[25] = {21, 1, 0, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
static const casadi_int casadi_s6[47] = {21, 21, 0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10};
static const casadi_int casadi_s7[24] = {0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* drone_ode_cost_ext_cost_fun_jac_hess:(i0[17],i1[4],i2[],i3[27])->(o0,o1[21],o2[21x21,23nz],o3[],o4[0x21]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=arg[0]? arg[0][0] : 0;
  a1=arg[3]? arg[3][14] : 0;
  a0=(a0-a1);
  a1=casadi_sq(a0);
  a2=arg[0]? arg[0][1] : 0;
  a3=arg[3]? arg[3][15] : 0;
  a2=(a2-a3);
  a3=casadi_sq(a2);
  a1=(a1+a3);
  a3=arg[0]? arg[0][2] : 0;
  a4=arg[3]? arg[3][16] : 0;
  a3=(a3-a4);
  a4=casadi_sq(a3);
  a1=(a1+a4);
  a4=5000.;
  a5=2.;
  a6=arg[0]? arg[0][5] : 0;
  a7=arg[0]? arg[0][3] : 0;
  a8=(a6+a7);
  a9=casadi_sq(a8);
  a10=arg[0]? arg[0][6] : 0;
  a11=arg[0]? arg[0][4] : 0;
  a12=(a10-a11);
  a13=casadi_sq(a12);
  a9=(a9+a13);
  a13=sqrt(a9);
  a14=(a7-a6);
  a15=casadi_sq(a14);
  a16=(a11+a10);
  a17=casadi_sq(a16);
  a15=(a15+a17);
  a17=sqrt(a15);
  a18=atan2(a13,a17);
  a18=(a5*a18);
  a19=casadi_fabs(a18);
  a20=9.9999999999999995e-08;
  a19=(a19<=a20);
  a21=1.;
  a22=(a19?a21:0);
  a19=(!a19);
  a23=3.1415926535897931e+00;
  a24=(a18-a23);
  a24=casadi_fabs(a24);
  a24=(a24<=a20);
  a24=(a24?a5:0);
  a19=(a19?a24:0);
  a22=(a22+a19);
  a19=0.;
  a24=(a22==a19);
  a25=atan2(a16,a14);
  a26=atan2(a12,a8);
  a27=(a25-a26);
  a27=(a24?a27:0);
  a28=(!a24);
  a22=(a22==a21);
  a29=(a5*a25);
  a29=(a22?a29:0);
  a30=(!a22);
  a31=(a5*a26);
  a31=(-a31);
  a31=(a30?a31:0);
  a29=(a29+a31);
  a29=(a28?a29:0);
  a27=(a27+a29);
  a29=-3.1415926535897931e+00;
  a31=(a27<a29);
  a32=6.2831853071795862e+00;
  a33=(a31?a32:0);
  a31=(!a31);
  a34=(a23<a27);
  a35=-6.2831853071795862e+00;
  a34=(a34?a35:0);
  a31=(a31?a34:0);
  a33=(a33+a31);
  a27=(a27+a33);
  a33=arg[3]? arg[3][19] : 0;
  a31=arg[3]? arg[3][17] : 0;
  a34=(a33+a31);
  a36=casadi_sq(a34);
  a37=arg[3]? arg[3][20] : 0;
  a38=arg[3]? arg[3][18] : 0;
  a39=(a37-a38);
  a40=casadi_sq(a39);
  a36=(a36+a40);
  a36=sqrt(a36);
  a40=(a31-a33);
  a41=casadi_sq(a40);
  a42=(a38+a37);
  a43=casadi_sq(a42);
  a41=(a41+a43);
  a41=sqrt(a41);
  a36=atan2(a36,a41);
  a36=(a5*a36);
  a41=casadi_fabs(a36);
  a41=(a41<=a20);
  a43=(a41?a21:0);
  a41=(!a41);
  a44=(a36-a23);
  a44=casadi_fabs(a44);
  a44=(a44<=a20);
  a44=(a44?a5:0);
  a41=(a41?a44:0);
  a43=(a43+a41);
  a41=(a43==a19);
  a42=atan2(a42,a40);
  a39=atan2(a39,a34);
  a34=(a42-a39);
  a34=(a41?a34:0);
  a40=(!a41);
  a43=(a43==a21);
  a44=(a5*a42);
  a44=(a43?a44:0);
  a43=(!a43);
  a45=(a5*a39);
  a45=(-a45);
  a43=(a43?a45:0);
  a44=(a44+a43);
  a44=(a40?a44:0);
  a34=(a34+a44);
  a44=(a34<a29);
  a43=(a44?a32:0);
  a44=(!a44);
  a45=(a23<a34);
  a45=(a45?a35:0);
  a44=(a44?a45:0);
  a43=(a43+a44);
  a34=(a34+a43);
  a27=(a27-a34);
  a27=(a4*a27);
  a34=(a6+a7);
  a43=casadi_sq(a34);
  a44=(a10-a11);
  a45=casadi_sq(a44);
  a43=(a43+a45);
  a45=sqrt(a43);
  a7=(a7-a6);
  a6=casadi_sq(a7);
  a11=(a11+a10);
  a10=casadi_sq(a11);
  a6=(a6+a10);
  a10=sqrt(a6);
  a46=atan2(a45,a10);
  a46=(a5*a46);
  a47=casadi_fabs(a46);
  a47=(a47<=a20);
  a48=(a47?a21:0);
  a47=(!a47);
  a49=(a46-a23);
  a49=casadi_fabs(a49);
  a49=(a49<=a20);
  a49=(a49?a5:0);
  a47=(a47?a49:0);
  a48=(a48+a47);
  a47=(a48==a19);
  a49=atan2(a11,a7);
  a50=atan2(a44,a34);
  a51=(a49-a50);
  a51=(a47?a51:0);
  a52=(!a47);
  a48=(a48==a21);
  a53=(a5*a49);
  a53=(a48?a53:0);
  a54=(!a48);
  a55=(a5*a50);
  a55=(-a55);
  a55=(a54?a55:0);
  a53=(a53+a55);
  a53=(a52?a53:0);
  a51=(a51+a53);
  a53=(a51<a29);
  a55=(a53?a32:0);
  a53=(!a53);
  a56=(a23<a51);
  a56=(a56?a35:0);
  a53=(a53?a56:0);
  a55=(a55+a53);
  a51=(a51+a55);
  a55=(a33+a31);
  a53=casadi_sq(a55);
  a56=(a37-a38);
  a57=casadi_sq(a56);
  a53=(a53+a57);
  a53=sqrt(a53);
  a31=(a31-a33);
  a33=casadi_sq(a31);
  a38=(a38+a37);
  a37=casadi_sq(a38);
  a33=(a33+a37);
  a33=sqrt(a33);
  a53=atan2(a53,a33);
  a53=(a5*a53);
  a33=casadi_fabs(a53);
  a33=(a33<=a20);
  a37=(a33?a21:0);
  a33=(!a33);
  a57=(a53-a23);
  a57=casadi_fabs(a57);
  a57=(a57<=a20);
  a57=(a57?a5:0);
  a33=(a33?a57:0);
  a37=(a37+a33);
  a33=(a37==a19);
  a38=atan2(a38,a31);
  a56=atan2(a56,a55);
  a55=(a38-a56);
  a55=(a33?a55:0);
  a31=(!a33);
  a37=(a37==a21);
  a21=(a5*a38);
  a21=(a37?a21:0);
  a37=(!a37);
  a57=(a5*a56);
  a57=(-a57);
  a37=(a37?a57:0);
  a21=(a21+a37);
  a21=(a31?a21:0);
  a55=(a55+a21);
  a21=(a55<a29);
  a37=(a21?a32:0);
  a21=(!a21);
  a57=(a23<a55);
  a57=(a57?a35:0);
  a21=(a21?a57:0);
  a37=(a37+a21);
  a55=(a55+a37);
  a51=(a51-a55);
  a55=(a27*a51);
  a1=(a1+a55);
  a55=(a24?a18:0);
  a18=(a28?a18:0);
  a55=(a55+a18);
  a18=1.5707963267948966e+00;
  a55=(a55-a18);
  a37=(a55<a29);
  a21=(a37?a32:0);
  a37=(!a37);
  a57=(a23<a55);
  a57=(a57?a35:0);
  a37=(a37?a57:0);
  a21=(a21+a37);
  a55=(a55+a21);
  a21=(a41?a36:0);
  a40=(a40?a36:0);
  a21=(a21+a40);
  a21=(a21-a18);
  a40=(a21<a29);
  a36=(a40?a32:0);
  a40=(!a40);
  a37=(a23<a21);
  a37=(a37?a35:0);
  a40=(a40?a37:0);
  a36=(a36+a40);
  a21=(a21+a36);
  a55=(a55-a21);
  a55=(a4*a55);
  a21=(a47?a46:0);
  a46=(a52?a46:0);
  a21=(a21+a46);
  a21=(a21-a18);
  a46=(a21<a29);
  a36=(a46?a32:0);
  a46=(!a46);
  a40=(a23<a21);
  a40=(a40?a35:0);
  a46=(a46?a40:0);
  a36=(a36+a46);
  a21=(a21+a36);
  a36=(a33?a53:0);
  a31=(a31?a53:0);
  a36=(a36+a31);
  a36=(a36-a18);
  a18=(a36<a29);
  a31=(a18?a32:0);
  a18=(!a18);
  a53=(a23<a36);
  a53=(a53?a35:0);
  a18=(a18?a53:0);
  a31=(a31+a18);
  a36=(a36+a31);
  a21=(a21-a36);
  a36=(a55*a21);
  a1=(a1+a36);
  a25=(a25+a26);
  a25=(a24?a25:0);
  a26=(a25<a29);
  a36=(a26?a32:0);
  a26=(!a26);
  a31=(a23<a25);
  a31=(a31?a35:0);
  a26=(a26?a31:0);
  a36=(a36+a26);
  a25=(a25+a36);
  a42=(a42+a39);
  a41=(a41?a42:0);
  a42=(a41<a29);
  a39=(a42?a32:0);
  a42=(!a42);
  a36=(a23<a41);
  a36=(a36?a35:0);
  a42=(a42?a36:0);
  a39=(a39+a42);
  a41=(a41+a39);
  a25=(a25-a41);
  a25=(a4*a25);
  a49=(a49+a50);
  a49=(a47?a49:0);
  a50=(a49<a29);
  a41=(a50?a32:0);
  a50=(!a50);
  a39=(a23<a49);
  a39=(a39?a35:0);
  a50=(a50?a39:0);
  a41=(a41+a50);
  a49=(a49+a41);
  a38=(a38+a56);
  a33=(a33?a38:0);
  a29=(a33<a29);
  a32=(a29?a32:0);
  a29=(!a29);
  a23=(a23<a33);
  a23=(a23?a35:0);
  a29=(a29?a23:0);
  a32=(a32+a29);
  a33=(a33+a32);
  a49=(a49-a33);
  a33=(a25*a49);
  a1=(a1+a33);
  a33=arg[1]? arg[1][0] : 0;
  a32=casadi_sq(a33);
  a29=arg[1]? arg[1][1] : 0;
  a23=casadi_sq(a29);
  a32=(a32+a23);
  a23=arg[1]? arg[1][2] : 0;
  a35=casadi_sq(a23);
  a32=(a32+a35);
  a35=arg[1]? arg[1][3] : 0;
  a38=casadi_sq(a35);
  a32=(a32+a38);
  a1=(a1+a32);
  if (res[0]!=0) res[0][0]=a1;
  a33=(a33+a33);
  if (res[1]!=0) res[1][0]=a33;
  a29=(a29+a29);
  if (res[1]!=0) res[1][1]=a29;
  a23=(a23+a23);
  if (res[1]!=0) res[1][2]=a23;
  a35=(a35+a35);
  if (res[1]!=0) res[1][3]=a35;
  a0=(a0+a0);
  if (res[1]!=0) res[1][4]=a0;
  a2=(a2+a2);
  if (res[1]!=0) res[1][5]=a2;
  a3=(a3+a3);
  if (res[1]!=0) res[1][6]=a3;
  a3=(a34+a34);
  a2=(a43+a6);
  a0=(a10/a2);
  a35=(a52?a55:0);
  a55=(a47?a55:0);
  a35=(a35+a55);
  a35=(a5*a35);
  a55=(a0*a35);
  a23=(a45+a45);
  a55=(a55/a23);
  a29=(a3*a55);
  a33=casadi_sq(a44);
  a1=casadi_sq(a34);
  a33=(a33+a1);
  a1=(a44/a33);
  a32=(a47?a25:0);
  a38=(a5*a27);
  a38=(-a38);
  a38=(a52?a38:0);
  a38=(a54?a38:0);
  a32=(a32+a38);
  a38=(-a27);
  a38=(a47?a38:0);
  a32=(a32+a38);
  a38=(a1*a32);
  a29=(a29-a38);
  a38=casadi_sq(a11);
  a56=casadi_sq(a7);
  a38=(a38+a56);
  a56=(a11/a38);
  a25=(a47?a25:0);
  a41=(a5*a27);
  a41=(a52?a41:0);
  a41=(a48?a41:0);
  a25=(a25+a41);
  a27=(a47?a27:0);
  a25=(a25+a27);
  a27=(a56*a25);
  a41=(a7+a7);
  a50=(a45/a2);
  a39=(a50*a35);
  a42=(a10+a10);
  a39=(a39/a42);
  a36=(a41*a39);
  a27=(a27+a36);
  a36=(a29-a27);
  a26=casadi_sq(a16);
  a31=casadi_sq(a14);
  a26=(a26+a31);
  a31=(a16/a26);
  a49=(a4*a49);
  a18=(a24?a49:0);
  a51=(a4*a51);
  a53=(a5*a51);
  a53=(a28?a53:0);
  a53=(a22?a53:0);
  a18=(a18+a53);
  a53=(a24?a51:0);
  a18=(a18+a53);
  a53=(a31*a18);
  a46=(a14+a14);
  a40=(a9+a15);
  a37=(a13/a40);
  a21=(a4*a21);
  a57=(a28?a21:0);
  a21=(a24?a21:0);
  a57=(a57+a21);
  a57=(a5*a57);
  a21=(a37*a57);
  a20=(a17+a17);
  a21=(a21/a20);
  a58=(a46*a21);
  a53=(a53+a58);
  a36=(a36-a53);
  a58=(a8+a8);
  a59=(a17/a40);
  a60=(a59*a57);
  a61=(a13+a13);
  a60=(a60/a61);
  a62=(a58*a60);
  a63=casadi_sq(a12);
  a64=casadi_sq(a8);
  a63=(a63+a64);
  a64=(a12/a63);
  a49=(a24?a49:0);
  a65=(a5*a51);
  a65=(-a65);
  a65=(a28?a65:0);
  a65=(a30?a65:0);
  a49=(a49+a65);
  a51=(-a51);
  a51=(a24?a51:0);
  a49=(a49+a51);
  a51=(a64*a49);
  a62=(a62-a51);
  a36=(a36+a62);
  if (res[1]!=0) res[1][7]=a36;
  a36=(a7/a38);
  a51=(a36*a25);
  a65=(a11+a11);
  a66=(a65*a39);
  a51=(a51-a66);
  a66=(a34/a33);
  a67=(a66*a32);
  a68=(a44+a44);
  a69=(a68*a55);
  a67=(a67+a69);
  a69=(a51-a67);
  a70=(a14/a26);
  a71=(a70*a18);
  a72=(a16+a16);
  a73=(a72*a21);
  a71=(a71-a73);
  a69=(a69+a71);
  a73=(a8/a63);
  a74=(a73*a49);
  a75=(a12+a12);
  a76=(a75*a60);
  a74=(a74+a76);
  a69=(a69-a74);
  if (res[1]!=0) res[1][8]=a69;
  a27=(a27+a29);
  a27=(a27+a53);
  a27=(a27+a62);
  if (res[1]!=0) res[1][9]=a27;
  a51=(a51+a67);
  a51=(a51+a71);
  a51=(a51+a74);
  if (res[1]!=0) res[1][10]=a51;
  if (res[1]!=0) res[1][11]=a19;
  if (res[1]!=0) res[1][12]=a19;
  if (res[1]!=0) res[1][13]=a19;
  if (res[1]!=0) res[1][14]=a19;
  if (res[1]!=0) res[1][15]=a19;
  if (res[1]!=0) res[1][16]=a19;
  if (res[1]!=0) res[1][17]=a19;
  if (res[1]!=0) res[1][18]=a19;
  if (res[1]!=0) res[1][19]=a19;
  if (res[1]!=0) res[1][20]=a19;
  if (res[2]!=0) res[2][0]=a5;
  if (res[2]!=0) res[2][1]=a5;
  if (res[2]!=0) res[2][2]=a5;
  if (res[2]!=0) res[2][3]=a5;
  if (res[2]!=0) res[2][4]=a5;
  if (res[2]!=0) res[2][5]=a5;
  if (res[2]!=0) res[2][6]=a5;
  a19=(a5*a55);
  a51=(a7/a10);
  a74=(a51/a2);
  a71=(a0/a2);
  a67=(a34+a34);
  a27=(a7+a7);
  a62=(a67+a27);
  a53=(a71*a62);
  a74=(a74-a53);
  a74=(a35*a74);
  a9=(a9+a15);
  a15=(a17/a9);
  a53=(a8/a13);
  a29=(a15*a53);
  a9=(a13/a9);
  a69=(a14/a17);
  a76=(a9*a69);
  a29=(a29-a76);
  a29=(a5*a29);
  a76=(a24?a29:0);
  a29=(a28?a29:0);
  a76=(a76+a29);
  a76=(a4*a76);
  a29=(a52?a76:0);
  a76=(a47?a76:0);
  a29=(a29+a76);
  a29=(a5*a29);
  a76=(a0*a29);
  a74=(a74+a76);
  a74=(a74/a23);
  a76=(a55/a23);
  a77=(a34/a45);
  a78=(a77+a77);
  a78=(a76*a78);
  a74=(a74-a78);
  a74=(a3*a74);
  a19=(a19+a74);
  a74=casadi_sq(a16);
  a78=casadi_sq(a14);
  a74=(a74+a78);
  a78=(a16/a74);
  a79=casadi_sq(a12);
  a80=casadi_sq(a8);
  a79=(a79+a80);
  a80=(a12/a79);
  a81=(a78+a80);
  a81=(a4*a81);
  a81=(-a81);
  a82=(a24?a81:0);
  a82=(a47?a82:0);
  a83=(a80-a78);
  a83=(a24?a83:0);
  a84=(a5*a78);
  a84=(-a84);
  a84=(a22?a84:0);
  a85=(a5*a80);
  a85=(a30?a85:0);
  a84=(a84+a85);
  a84=(a28?a84:0);
  a83=(a83+a84);
  a83=(a4*a83);
  a84=(a5*a83);
  a84=(-a84);
  a84=(a52?a84:0);
  a84=(a54?a84:0);
  a82=(a82+a84);
  a84=(-a83);
  a84=(a47?a84:0);
  a82=(a82+a84);
  a82=(a1*a82);
  a84=(a1/a33);
  a85=(a34+a34);
  a86=(a84*a85);
  a86=(a32*a86);
  a82=(a82-a86);
  a19=(a19-a82);
  a81=(a24?a81:0);
  a81=(a47?a81:0);
  a82=(a5*a83);
  a82=(a52?a82:0);
  a82=(a48?a82:0);
  a81=(a81+a82);
  a83=(a47?a83:0);
  a81=(a81+a83);
  a81=(a56*a81);
  a83=(a56/a38);
  a82=(a7+a7);
  a86=(a83*a82);
  a86=(a25*a86);
  a81=(a81-a86);
  a86=(a5*a39);
  a87=(a77/a2);
  a88=(a50/a2);
  a62=(a88*a62);
  a87=(a87-a62);
  a87=(a35*a87);
  a29=(a50*a29);
  a87=(a87+a29);
  a87=(a87/a42);
  a29=(a39/a42);
  a62=(a51+a51);
  a62=(a29*a62);
  a87=(a87-a62);
  a87=(a41*a87);
  a86=(a86+a87);
  a81=(a81+a86);
  a19=(a19-a81);
  a81=casadi_sq(a11);
  a86=casadi_sq(a7);
  a81=(a81+a86);
  a86=(a11/a81);
  a87=casadi_sq(a44);
  a62=casadi_sq(a34);
  a87=(a87+a62);
  a62=(a44/a87);
  a89=(a86+a62);
  a89=(a4*a89);
  a89=(-a89);
  a90=(a47?a89:0);
  a90=(a24?a90:0);
  a91=(a62-a86);
  a91=(a47?a91:0);
  a92=(a5*a86);
  a92=(-a92);
  a92=(a48?a92:0);
  a93=(a5*a62);
  a93=(a54?a93:0);
  a92=(a92+a93);
  a92=(a52?a92:0);
  a91=(a91+a92);
  a91=(a4*a91);
  a92=(a5*a91);
  a92=(a28?a92:0);
  a92=(a22?a92:0);
  a90=(a90+a92);
  a92=(a24?a91:0);
  a90=(a90+a92);
  a90=(a31*a90);
  a92=(a31/a26);
  a93=(a14+a14);
  a94=(a92*a93);
  a94=(a18*a94);
  a90=(a90-a94);
  a94=(a5*a21);
  a95=(a53/a40);
  a96=(a37/a40);
  a97=(a8+a8);
  a98=(a14+a14);
  a99=(a97+a98);
  a100=(a96*a99);
  a95=(a95-a100);
  a95=(a57*a95);
  a43=(a43+a6);
  a6=(a10/a43);
  a77=(a6*a77);
  a43=(a45/a43);
  a51=(a43*a51);
  a77=(a77-a51);
  a77=(a5*a77);
  a51=(a47?a77:0);
  a77=(a52?a77:0);
  a51=(a51+a77);
  a51=(a4*a51);
  a77=(a28?a51:0);
  a51=(a24?a51:0);
  a77=(a77+a51);
  a77=(a5*a77);
  a51=(a37*a77);
  a95=(a95+a51);
  a95=(a95/a20);
  a51=(a21/a20);
  a100=(a69+a69);
  a100=(a51*a100);
  a95=(a95-a100);
  a95=(a46*a95);
  a94=(a94+a95);
  a90=(a90+a94);
  a19=(a19-a90);
  a90=(a5*a60);
  a69=(a69/a40);
  a94=(a59/a40);
  a99=(a94*a99);
  a69=(a69-a99);
  a69=(a57*a69);
  a77=(a59*a77);
  a69=(a69+a77);
  a69=(a69/a61);
  a77=(a60/a61);
  a53=(a53+a53);
  a53=(a77*a53);
  a69=(a69-a53);
  a69=(a58*a69);
  a90=(a90+a69);
  a89=(a47?a89:0);
  a89=(a24?a89:0);
  a69=(a5*a91);
  a69=(-a69);
  a69=(a28?a69:0);
  a69=(a30?a69:0);
  a89=(a89+a69);
  a91=(-a91);
  a91=(a24?a91:0);
  a89=(a89+a91);
  a89=(a64*a89);
  a91=(a64/a63);
  a69=(a8+a8);
  a53=(a91*a69);
  a53=(a49*a53);
  a89=(a89-a53);
  a90=(a90-a89);
  a19=(a19+a90);
  if (res[2]!=0) res[2][7]=a19;
  a19=(a11/a10);
  a90=(a19/a2);
  a89=(a11+a11);
  a53=(a44+a44);
  a99=(a89-a53);
  a95=(a71*a99);
  a90=(a90-a95);
  a90=(a35*a90);
  a95=(a12/a13);
  a100=(a15*a95);
  a101=(a16/a17);
  a102=(a9*a101);
  a100=(a100+a102);
  a100=(a5*a100);
  a100=(-a100);
  a102=(a24?a100:0);
  a100=(a28?a100:0);
  a102=(a102+a100);
  a102=(a4*a102);
  a100=(a52?a102:0);
  a102=(a47?a102:0);
  a100=(a100+a102);
  a100=(a5*a100);
  a102=(a0*a100);
  a90=(a90+a102);
  a90=(a90/a23);
  a102=(a44/a45);
  a103=(a102+a102);
  a103=(a76*a103);
  a90=(a90+a103);
  a103=(a3*a90);
  a104=(a44+a44);
  a105=(a84*a104);
  a106=(1./a33);
  a105=(a105-a106);
  a105=(a32*a105);
  a74=(a14/a74);
  a79=(a8/a79);
  a107=(a74-a79);
  a107=(a4*a107);
  a108=(a24?a107:0);
  a108=(a47?a108:0);
  a109=(a74+a79);
  a109=(a24?a109:0);
  a110=(a5*a74);
  a110=(a22?a110:0);
  a111=(a5*a79);
  a111=(a30?a111:0);
  a110=(a110+a111);
  a110=(a28?a110:0);
  a109=(a109+a110);
  a109=(a4*a109);
  a110=(a5*a109);
  a110=(-a110);
  a110=(a52?a110:0);
  a110=(a54?a110:0);
  a108=(a108+a110);
  a110=(-a109);
  a110=(a47?a110:0);
  a108=(a108+a110);
  a110=(a1*a108);
  a105=(a105+a110);
  a103=(a103-a105);
  a105=(1./a38);
  a110=(a11+a11);
  a111=(a83*a110);
  a111=(a105-a111);
  a111=(a25*a111);
  a107=(a24?a107:0);
  a107=(a47?a107:0);
  a112=(a5*a109);
  a112=(a52?a112:0);
  a112=(a48?a112:0);
  a107=(a107+a112);
  a109=(a47?a109:0);
  a107=(a107+a109);
  a109=(a56*a107);
  a111=(a111+a109);
  a100=(a50*a100);
  a109=(a102/a2);
  a99=(a88*a99);
  a109=(a109+a99);
  a109=(a35*a109);
  a100=(a100-a109);
  a100=(a100/a42);
  a109=(a19+a19);
  a109=(a29*a109);
  a100=(a100-a109);
  a109=(a41*a100);
  a111=(a111+a109);
  a103=(a103-a111);
  a111=(1./a26);
  a109=(a16+a16);
  a99=(a92*a109);
  a99=(a111-a99);
  a99=(a18*a99);
  a81=(a7/a81);
  a87=(a34/a87);
  a112=(a81-a87);
  a112=(a4*a112);
  a113=(a47?a112:0);
  a113=(a24?a113:0);
  a114=(a81+a87);
  a114=(a47?a114:0);
  a115=(a5*a81);
  a115=(a48?a115:0);
  a116=(a5*a87);
  a116=(a54?a116:0);
  a115=(a115+a116);
  a115=(a52?a115:0);
  a114=(a114+a115);
  a114=(a4*a114);
  a115=(a5*a114);
  a115=(a28?a115:0);
  a115=(a22?a115:0);
  a113=(a113+a115);
  a115=(a24?a114:0);
  a113=(a113+a115);
  a115=(a31*a113);
  a99=(a99+a115);
  a102=(a6*a102);
  a19=(a43*a19);
  a102=(a102+a19);
  a102=(a5*a102);
  a102=(-a102);
  a19=(a47?a102:0);
  a102=(a52?a102:0);
  a19=(a19+a102);
  a19=(a4*a19);
  a102=(a28?a19:0);
  a19=(a24?a19:0);
  a102=(a102+a19);
  a102=(a5*a102);
  a19=(a37*a102);
  a115=(a95/a40);
  a116=(a16+a16);
  a117=(a12+a12);
  a118=(a116-a117);
  a119=(a96*a118);
  a115=(a115+a119);
  a115=(a57*a115);
  a19=(a19-a115);
  a19=(a19/a20);
  a115=(a101+a101);
  a115=(a51*a115);
  a19=(a19-a115);
  a115=(a46*a19);
  a99=(a99+a115);
  a103=(a103-a99);
  a101=(a101/a40);
  a118=(a94*a118);
  a101=(a101-a118);
  a101=(a57*a101);
  a102=(a59*a102);
  a101=(a101+a102);
  a101=(a101/a61);
  a95=(a95+a95);
  a95=(a77*a95);
  a101=(a101+a95);
  a95=(a58*a101);
  a102=(a12+a12);
  a118=(a91*a102);
  a99=(1./a63);
  a118=(a118-a99);
  a118=(a49*a118);
  a112=(a47?a112:0);
  a112=(a24?a112:0);
  a115=(a5*a114);
  a115=(-a115);
  a115=(a28?a115:0);
  a115=(a30?a115:0);
  a112=(a112+a115);
  a114=(-a114);
  a114=(a24?a114:0);
  a112=(a112+a114);
  a114=(a64*a112);
  a118=(a118+a114);
  a95=(a95-a118);
  a103=(a103+a95);
  if (res[2]!=0) res[2][8]=a103;
  a95=(a5*a55);
  a8=(a8/a13);
  a118=(a15*a8);
  a14=(a14/a17);
  a114=(a9*a14);
  a118=(a118+a114);
  a118=(a5*a118);
  a114=(a24?a118:0);
  a118=(a28?a118:0);
  a114=(a114+a118);
  a114=(a4*a114);
  a118=(a52?a114:0);
  a114=(a47?a114:0);
  a118=(a118+a114);
  a118=(a5*a118);
  a114=(a0*a118);
  a7=(a7/a10);
  a115=(a7/a2);
  a67=(a67-a27);
  a27=(a71*a67);
  a115=(a115+a27);
  a115=(a35*a115);
  a114=(a114-a115);
  a114=(a114/a23);
  a34=(a34/a45);
  a115=(a34+a34);
  a115=(a76*a115);
  a114=(a114-a115);
  a115=(a3*a114);
  a95=(a95+a115);
  a115=(a78-a80);
  a115=(a4*a115);
  a27=(a24?a115:0);
  a27=(a47?a27:0);
  a119=(a78+a80);
  a119=(a24?a119:0);
  a78=(a5*a78);
  a78=(a22?a78:0);
  a80=(a5*a80);
  a80=(a30?a80:0);
  a78=(a78+a80);
  a78=(a28?a78:0);
  a119=(a119+a78);
  a119=(a4*a119);
  a78=(a5*a119);
  a78=(-a78);
  a78=(a52?a78:0);
  a78=(a54?a78:0);
  a27=(a27+a78);
  a78=(-a119);
  a78=(a47?a78:0);
  a27=(a27+a78);
  a78=(a1*a27);
  a80=(a84*a85);
  a80=(a32*a80);
  a78=(a78-a80);
  a95=(a95-a78);
  a78=(a83*a82);
  a78=(a25*a78);
  a115=(a24?a115:0);
  a115=(a47?a115:0);
  a80=(a5*a119);
  a80=(a52?a80:0);
  a80=(a48?a80:0);
  a115=(a115+a80);
  a119=(a47?a119:0);
  a115=(a115+a119);
  a119=(a56*a115);
  a78=(a78+a119);
  a119=-2.;
  a80=(a119*a39);
  a120=(a34/a2);
  a67=(a88*a67);
  a120=(a120-a67);
  a120=(a35*a120);
  a118=(a50*a118);
  a120=(a120+a118);
  a120=(a120/a42);
  a118=(a7+a7);
  a118=(a29*a118);
  a120=(a120+a118);
  a118=(a41*a120);
  a80=(a80+a118);
  a78=(a78+a80);
  a80=(a95-a78);
  a118=(a92*a93);
  a118=(a18*a118);
  a67=(a86-a62);
  a67=(a4*a67);
  a121=(a47?a67:0);
  a121=(a24?a121:0);
  a122=(a86+a62);
  a122=(a47?a122:0);
  a86=(a5*a86);
  a86=(a48?a86:0);
  a62=(a5*a62);
  a62=(a54?a62:0);
  a86=(a86+a62);
  a86=(a52?a86:0);
  a122=(a122+a86);
  a122=(a4*a122);
  a86=(a5*a122);
  a86=(a28?a86:0);
  a86=(a22?a86:0);
  a121=(a121+a86);
  a86=(a24?a122:0);
  a121=(a121+a86);
  a86=(a31*a121);
  a118=(a118+a86);
  a86=(a119*a21);
  a62=(a8/a40);
  a97=(a97-a98);
  a98=(a96*a97);
  a62=(a62-a98);
  a62=(a57*a62);
  a34=(a6*a34);
  a7=(a43*a7);
  a34=(a34+a7);
  a34=(a5*a34);
  a7=(a47?a34:0);
  a34=(a52?a34:0);
  a7=(a7+a34);
  a7=(a4*a7);
  a34=(a28?a7:0);
  a7=(a24?a7:0);
  a34=(a34+a7);
  a34=(a5*a34);
  a7=(a37*a34);
  a62=(a62+a7);
  a62=(a62/a20);
  a7=(a14+a14);
  a7=(a51*a7);
  a62=(a62+a7);
  a7=(a46*a62);
  a86=(a86+a7);
  a118=(a118+a86);
  a80=(a80-a118);
  a86=(a5*a60);
  a34=(a59*a34);
  a14=(a14/a40);
  a97=(a94*a97);
  a14=(a14+a97);
  a14=(a57*a14);
  a34=(a34-a14);
  a34=(a34/a61);
  a8=(a8+a8);
  a8=(a77*a8);
  a34=(a34-a8);
  a8=(a58*a34);
  a86=(a86+a8);
  a67=(a47?a67:0);
  a67=(a24?a67:0);
  a8=(a5*a122);
  a8=(-a8);
  a8=(a28?a8:0);
  a8=(a30?a8:0);
  a67=(a67+a8);
  a122=(-a122);
  a122=(a24?a122:0);
  a67=(a67+a122);
  a122=(a64*a67);
  a8=(a91*a69);
  a8=(a49*a8);
  a122=(a122-a8);
  a86=(a86-a122);
  a80=(a80+a86);
  if (res[2]!=0) res[2][9]=a80;
  a11=(a11/a10);
  a10=(a11/a2);
  a53=(a53+a89);
  a71=(a71*a53);
  a10=(a10-a71);
  a10=(a35*a10);
  a12=(a12/a13);
  a15=(a15*a12);
  a16=(a16/a17);
  a9=(a9*a16);
  a15=(a15-a9);
  a15=(a5*a15);
  a9=(a24?a15:0);
  a15=(a28?a15:0);
  a9=(a9+a15);
  a9=(a4*a9);
  a15=(a52?a9:0);
  a9=(a47?a9:0);
  a15=(a15+a9);
  a15=(a5*a15);
  a0=(a0*a15);
  a10=(a10+a0);
  a10=(a10/a23);
  a44=(a44/a45);
  a45=(a44+a44);
  a76=(a76*a45);
  a10=(a10-a76);
  a3=(a3*a10);
  a84=(a84*a104);
  a106=(a106-a84);
  a106=(a32*a106);
  a84=(a74+a79);
  a84=(a4*a84);
  a76=(a24?a84:0);
  a76=(a47?a76:0);
  a45=(a74-a79);
  a45=(a24?a45:0);
  a74=(a5*a74);
  a74=(a22?a74:0);
  a79=(a5*a79);
  a79=(-a79);
  a79=(a30?a79:0);
  a74=(a74+a79);
  a74=(a28?a74:0);
  a45=(a45+a74);
  a45=(a4*a45);
  a74=(a5*a45);
  a74=(-a74);
  a74=(a52?a74:0);
  a74=(a54?a74:0);
  a76=(a76+a74);
  a74=(-a45);
  a74=(a47?a74:0);
  a76=(a76+a74);
  a1=(a1*a76);
  a106=(a106+a1);
  a3=(a3-a106);
  a83=(a83*a110);
  a105=(a105-a83);
  a105=(a25*a105);
  a84=(a24?a84:0);
  a84=(a47?a84:0);
  a83=(a5*a45);
  a83=(a52?a83:0);
  a83=(a48?a83:0);
  a84=(a84+a83);
  a45=(a47?a45:0);
  a84=(a84+a45);
  a56=(a56*a84);
  a105=(a105+a56);
  a2=(a44/a2);
  a88=(a88*a53);
  a2=(a2-a88);
  a35=(a35*a2);
  a50=(a50*a15);
  a35=(a35+a50);
  a35=(a35/a42);
  a42=(a11+a11);
  a29=(a29*a42);
  a35=(a35-a29);
  a41=(a41*a35);
  a105=(a105+a41);
  a41=(a3-a105);
  a92=(a92*a109);
  a111=(a111-a92);
  a111=(a18*a111);
  a92=(a81+a87);
  a92=(a4*a92);
  a29=(a47?a92:0);
  a29=(a24?a29:0);
  a42=(a81-a87);
  a42=(a47?a42:0);
  a81=(a5*a81);
  a48=(a48?a81:0);
  a87=(a5*a87);
  a87=(-a87);
  a54=(a54?a87:0);
  a48=(a48+a54);
  a48=(a52?a48:0);
  a42=(a42+a48);
  a42=(a4*a42);
  a48=(a5*a42);
  a48=(a28?a48:0);
  a22=(a22?a48:0);
  a29=(a29+a22);
  a22=(a24?a42:0);
  a29=(a29+a22);
  a31=(a31*a29);
  a111=(a111+a31);
  a31=(a12/a40);
  a117=(a117+a116);
  a96=(a96*a117);
  a31=(a31-a96);
  a31=(a57*a31);
  a6=(a6*a44);
  a43=(a43*a11);
  a6=(a6-a43);
  a6=(a5*a6);
  a43=(a47?a6:0);
  a52=(a52?a6:0);
  a43=(a43+a52);
  a4=(a4*a43);
  a43=(a28?a4:0);
  a4=(a24?a4:0);
  a43=(a43+a4);
  a43=(a5*a43);
  a37=(a37*a43);
  a31=(a31+a37);
  a31=(a31/a20);
  a20=(a16+a16);
  a51=(a51*a20);
  a31=(a31-a51);
  a46=(a46*a31);
  a111=(a111+a46);
  a41=(a41-a111);
  a16=(a16/a40);
  a94=(a94*a117);
  a16=(a16-a94);
  a57=(a57*a16);
  a59=(a59*a43);
  a57=(a57+a59);
  a57=(a57/a61);
  a12=(a12+a12);
  a77=(a77*a12);
  a57=(a57-a77);
  a58=(a58*a57);
  a91=(a91*a102);
  a99=(a99-a91);
  a99=(a49*a99);
  a47=(a47?a92:0);
  a47=(a24?a47:0);
  a92=(a5*a42);
  a92=(-a92);
  a28=(a28?a92:0);
  a30=(a30?a28:0);
  a47=(a47+a30);
  a42=(-a42);
  a24=(a24?a42:0);
  a47=(a47+a24);
  a64=(a64*a47);
  a99=(a99+a64);
  a58=(a58-a99);
  a41=(a41+a58);
  if (res[2]!=0) res[2][10]=a41;
  if (res[2]!=0) res[2][11]=a103;
  a107=(a36*a107);
  a103=(a36/a38);
  a99=(a103*a110);
  a99=(a25*a99);
  a107=(a107-a99);
  a99=(a5*a39);
  a100=(a65*a100);
  a99=(a99+a100);
  a107=(a107-a99);
  a99=(a66/a33);
  a100=(a99*a104);
  a100=(a32*a100);
  a108=(a66*a108);
  a100=(a100+a108);
  a108=(a119*a55);
  a90=(a68*a90);
  a108=(a108+a90);
  a100=(a100+a108);
  a107=(a107-a100);
  a113=(a70*a113);
  a100=(a70/a26);
  a108=(a100*a109);
  a108=(a18*a108);
  a113=(a113-a108);
  a108=(a5*a21);
  a19=(a72*a19);
  a108=(a108+a19);
  a113=(a113-a108);
  a107=(a107+a113);
  a113=(a73/a63);
  a108=(a113*a102);
  a108=(a49*a108);
  a112=(a73*a112);
  a108=(a108+a112);
  a119=(a119*a60);
  a101=(a75*a101);
  a119=(a119+a101);
  a108=(a108+a119);
  a107=(a107-a108);
  if (res[2]!=0) res[2][12]=a107;
  a82=(a103*a82);
  a38=(1./a38);
  a82=(a82-a38);
  a82=(a25*a82);
  a115=(a36*a115);
  a82=(a82+a115);
  a120=(a65*a120);
  a82=(a82-a120);
  a33=(1./a33);
  a85=(a99*a85);
  a33=(a33-a85);
  a33=(a32*a33);
  a27=(a66*a27);
  a33=(a33+a27);
  a114=(a68*a114);
  a33=(a33+a114);
  a82=(a82-a33);
  a93=(a100*a93);
  a26=(1./a26);
  a93=(a93-a26);
  a93=(a18*a93);
  a121=(a70*a121);
  a93=(a93+a121);
  a62=(a72*a62);
  a93=(a93-a62);
  a82=(a82+a93);
  a63=(1./a63);
  a69=(a113*a69);
  a63=(a63-a69);
  a63=(a49*a63);
  a67=(a73*a67);
  a63=(a63+a67);
  a34=(a75*a34);
  a63=(a63+a34);
  a82=(a82-a63);
  if (res[2]!=0) res[2][13]=a82;
  a36=(a36*a84);
  a103=(a103*a110);
  a25=(a25*a103);
  a36=(a36-a25);
  a39=(a5*a39);
  a65=(a65*a35);
  a39=(a39+a65);
  a36=(a36-a39);
  a66=(a66*a76);
  a99=(a99*a104);
  a32=(a32*a99);
  a66=(a66-a32);
  a55=(a5*a55);
  a68=(a68*a10);
  a55=(a55+a68);
  a66=(a66+a55);
  a55=(a36-a66);
  a70=(a70*a29);
  a100=(a100*a109);
  a18=(a18*a100);
  a70=(a70-a18);
  a21=(a5*a21);
  a72=(a72*a31);
  a21=(a21+a72);
  a70=(a70-a21);
  a55=(a55+a70);
  a73=(a73*a47);
  a113=(a113*a102);
  a49=(a49*a113);
  a73=(a73-a49);
  a5=(a5*a60);
  a75=(a75*a57);
  a5=(a5+a75);
  a73=(a73+a5);
  a55=(a55-a73);
  if (res[2]!=0) res[2][14]=a55;
  if (res[2]!=0) res[2][15]=a80;
  if (res[2]!=0) res[2][16]=a82;
  a78=(a78+a95);
  a78=(a78+a118);
  a78=(a78+a86);
  if (res[2]!=0) res[2][17]=a78;
  a105=(a105+a3);
  a105=(a105+a111);
  a105=(a105+a58);
  if (res[2]!=0) res[2][18]=a105;
  if (res[2]!=0) res[2][19]=a41;
  if (res[2]!=0) res[2][20]=a55;
  if (res[2]!=0) res[2][21]=a105;
  a36=(a36+a66);
  a36=(a36+a70);
  a36=(a36+a73);
  if (res[2]!=0) res[2][22]=a36;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_cost_ext_cost_fun_jac_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_fun_jac_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_fun_jac_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    case 3: return casadi_s2;
    case 4: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_jac_hess_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 5*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
