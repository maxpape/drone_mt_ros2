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
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
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
  a4=2.;
  a5=arg[0]? arg[0][3] : 0;
  a6=casadi_sq(a5);
  a7=arg[0]? arg[0][4] : 0;
  a8=casadi_sq(a7);
  a6=(a6+a8);
  a8=arg[0]? arg[0][5] : 0;
  a9=casadi_sq(a8);
  a6=(a6+a9);
  a9=arg[0]? arg[0][6] : 0;
  a10=casadi_sq(a9);
  a6=(a6+a10);
  a10=(a5/a6);
  a11=arg[3]? arg[3][18] : 0;
  a12=(a10*a11);
  a13=arg[3]? arg[3][17] : 0;
  a14=(a7/a6);
  a15=(a13*a14);
  a12=(a12-a15);
  a15=arg[3]? arg[3][20] : 0;
  a16=(a8/a6);
  a17=(a15*a16);
  a18=arg[3]? arg[3][19] : 0;
  a19=(a9/a6);
  a20=(a18*a19);
  a17=(a17-a20);
  a12=(a12+a17);
  a17=casadi_sq(a12);
  a20=(a10*a18);
  a21=(a13*a16);
  a20=(a20-a21);
  a21=(a11*a19);
  a22=(a15*a14);
  a21=(a21-a22);
  a20=(a20+a21);
  a21=casadi_sq(a20);
  a17=(a17+a21);
  a21=(a10*a15);
  a22=(a13*a19);
  a21=(a21-a22);
  a22=(a18*a14);
  a23=(a11*a16);
  a22=(a22-a23);
  a21=(a21+a22);
  a22=casadi_sq(a21);
  a17=(a17+a22);
  a22=sqrt(a17);
  a23=(a13*a10);
  a24=(a11*a14);
  a25=(a18*a16);
  a24=(a24+a25);
  a25=(a15*a19);
  a24=(a24+a25);
  a23=(a23+a24);
  a24=atan2(a22,a23);
  a24=(a4*a24);
  a25=casadi_sq(a5);
  a26=casadi_sq(a7);
  a25=(a25+a26);
  a26=casadi_sq(a8);
  a25=(a25+a26);
  a26=casadi_sq(a9);
  a25=(a25+a26);
  a26=(a5/a25);
  a27=(a26*a11);
  a28=(a7/a25);
  a29=(a13*a28);
  a27=(a27-a29);
  a29=(a8/a25);
  a30=(a15*a29);
  a31=(a9/a25);
  a32=(a18*a31);
  a30=(a30-a32);
  a27=(a27+a30);
  a30=casadi_sq(a27);
  a32=(a26*a18);
  a33=(a13*a29);
  a32=(a32-a33);
  a33=(a11*a31);
  a34=(a15*a28);
  a33=(a33-a34);
  a32=(a32+a33);
  a33=casadi_sq(a32);
  a30=(a30+a33);
  a33=(a26*a15);
  a34=(a13*a31);
  a33=(a33-a34);
  a34=(a18*a28);
  a35=(a11*a29);
  a34=(a34-a35);
  a33=(a33+a34);
  a34=casadi_sq(a33);
  a30=(a30+a34);
  a34=sqrt(a30);
  a35=(a13*a26);
  a36=(a11*a28);
  a37=(a18*a29);
  a36=(a36+a37);
  a37=(a15*a31);
  a36=(a36+a37);
  a35=(a35+a36);
  a36=atan2(a34,a35);
  a36=(a4*a36);
  a37=(a24*a36);
  a1=(a1+a37);
  a37=2.0000000000000001e-01;
  a38=arg[1]? arg[1][0] : 0;
  a39=(a37*a38);
  a40=(a39*a38);
  a41=arg[1]? arg[1][1] : 0;
  a42=(a37*a41);
  a43=(a42*a41);
  a40=(a40+a43);
  a43=arg[1]? arg[1][2] : 0;
  a44=(a37*a43);
  a45=(a44*a43);
  a40=(a40+a45);
  a45=arg[1]? arg[1][3] : 0;
  a46=(a37*a45);
  a47=(a46*a45);
  a40=(a40+a47);
  a1=(a1+a40);
  if (res[0]!=0) res[0][0]=a1;
  a38=(a37*a38);
  a39=(a39+a38);
  if (res[1]!=0) res[1][0]=a39;
  a41=(a37*a41);
  a42=(a42+a41);
  if (res[1]!=0) res[1][1]=a42;
  a43=(a37*a43);
  a44=(a44+a43);
  if (res[1]!=0) res[1][2]=a44;
  a37=(a37*a45);
  a46=(a46+a37);
  if (res[1]!=0) res[1][3]=a46;
  a0=(a0+a0);
  if (res[1]!=0) res[1][4]=a0;
  a2=(a2+a2);
  if (res[1]!=0) res[1][5]=a2;
  a3=(a3+a3);
  if (res[1]!=0) res[1][6]=a3;
  a3=(a33+a33);
  a2=casadi_sq(a35);
  a2=(a30+a2);
  a0=(a35/a2);
  a24=(a4*a24);
  a46=(a0*a24);
  a37=(a34+a34);
  a46=(a46/a37);
  a45=(a3*a46);
  a44=(a15*a45);
  a43=(a34/a2);
  a42=(a43*a24);
  a41=(a13*a42);
  a44=(a44-a41);
  a41=(a32+a32);
  a39=(a41*a46);
  a38=(a18*a39);
  a44=(a44+a38);
  a38=(a27+a27);
  a1=(a38*a46);
  a40=(a11*a1);
  a44=(a44+a40);
  a40=(a44/a25);
  a47=(a5+a5);
  a48=(a31/a25);
  a49=(a11*a39);
  a50=(a15*a42);
  a51=(a13*a45);
  a50=(a50+a51);
  a49=(a49-a50);
  a50=(a18*a1);
  a49=(a49-a50);
  a50=(a48*a49);
  a51=(a29/a25);
  a52=(a15*a1);
  a53=(a18*a42);
  a54=(a11*a45);
  a53=(a53+a54);
  a54=(a13*a39);
  a53=(a53+a54);
  a52=(a52-a53);
  a53=(a51*a52);
  a50=(a50+a53);
  a53=(a28/a25);
  a45=(a18*a45);
  a42=(a11*a42);
  a45=(a45-a42);
  a39=(a15*a39);
  a45=(a45-a39);
  a1=(a13*a1);
  a45=(a45-a1);
  a1=(a53*a45);
  a50=(a50+a1);
  a1=(a26/a25);
  a39=(a1*a44);
  a50=(a50+a39);
  a39=(a47*a50);
  a39=(a40-a39);
  a42=(a21+a21);
  a54=casadi_sq(a23);
  a54=(a17+a54);
  a55=(a23/a54);
  a36=(a4*a36);
  a56=(a55*a36);
  a57=(a22+a22);
  a56=(a56/a57);
  a58=(a42*a56);
  a59=(a15*a58);
  a60=(a22/a54);
  a61=(a60*a36);
  a62=(a13*a61);
  a59=(a59-a62);
  a62=(a20+a20);
  a63=(a62*a56);
  a64=(a18*a63);
  a59=(a59+a64);
  a64=(a12+a12);
  a65=(a64*a56);
  a66=(a11*a65);
  a59=(a59+a66);
  a66=(a59/a6);
  a39=(a39+a66);
  a67=(a5+a5);
  a68=(a19/a6);
  a69=(a11*a63);
  a70=(a15*a61);
  a71=(a13*a58);
  a70=(a70+a71);
  a69=(a69-a70);
  a70=(a18*a65);
  a69=(a69-a70);
  a70=(a68*a69);
  a71=(a16/a6);
  a72=(a15*a65);
  a73=(a18*a61);
  a74=(a11*a58);
  a73=(a73+a74);
  a74=(a13*a63);
  a73=(a73+a74);
  a72=(a72-a73);
  a73=(a71*a72);
  a70=(a70+a73);
  a73=(a14/a6);
  a58=(a18*a58);
  a61=(a11*a61);
  a58=(a58-a61);
  a63=(a15*a63);
  a58=(a58-a63);
  a65=(a13*a65);
  a58=(a58-a65);
  a65=(a73*a58);
  a70=(a70+a65);
  a65=(a10/a6);
  a63=(a65*a59);
  a70=(a70+a63);
  a63=(a67*a70);
  a39=(a39-a63);
  if (res[1]!=0) res[1][7]=a39;
  a39=(a45/a25);
  a63=(a7+a7);
  a61=(a63*a50);
  a61=(a39-a61);
  a74=(a58/a6);
  a61=(a61+a74);
  a75=(a7+a7);
  a76=(a75*a70);
  a61=(a61-a76);
  if (res[1]!=0) res[1][8]=a61;
  a61=(a52/a25);
  a76=(a8+a8);
  a77=(a76*a50);
  a77=(a61-a77);
  a78=(a72/a6);
  a77=(a77+a78);
  a79=(a8+a8);
  a80=(a79*a70);
  a77=(a77-a80);
  if (res[1]!=0) res[1][9]=a77;
  a77=(a49/a25);
  a80=(a9+a9);
  a81=(a80*a50);
  a81=(a77-a81);
  a82=(a69/a6);
  a81=(a81+a82);
  a83=(a9+a9);
  a84=(a83*a70);
  a81=(a81-a84);
  if (res[1]!=0) res[1][10]=a81;
  a81=0.;
  if (res[1]!=0) res[1][11]=a81;
  if (res[1]!=0) res[1][12]=a81;
  if (res[1]!=0) res[1][13]=a81;
  if (res[1]!=0) res[1][14]=a81;
  if (res[1]!=0) res[1][15]=a81;
  if (res[1]!=0) res[1][16]=a81;
  if (res[1]!=0) res[1][17]=a81;
  if (res[1]!=0) res[1][18]=a81;
  if (res[1]!=0) res[1][19]=a81;
  if (res[1]!=0) res[1][20]=a81;
  a81=4.0000000000000002e-01;
  if (res[2]!=0) res[2][0]=a81;
  if (res[2]!=0) res[2][1]=a81;
  if (res[2]!=0) res[2][2]=a81;
  if (res[2]!=0) res[2][3]=a81;
  if (res[2]!=0) res[2][4]=a4;
  if (res[2]!=0) res[2][5]=a4;
  if (res[2]!=0) res[2][6]=a4;
  a81=(1./a25);
  a26=(a26/a25);
  a84=(a5+a5);
  a85=(a26*a84);
  a81=(a81-a85);
  a85=(a15*a81);
  a31=(a31/a25);
  a86=(a31*a84);
  a87=(a13*a86);
  a85=(a85+a87);
  a29=(a29/a25);
  a87=(a29*a84);
  a88=(a11*a87);
  a28=(a28/a25);
  a89=(a28*a84);
  a90=(a18*a89);
  a88=(a88-a90);
  a85=(a85+a88);
  a88=(a85+a85);
  a88=(a46*a88);
  a90=(a13*a81);
  a91=(a11*a89);
  a92=(a18*a87);
  a91=(a91+a92);
  a92=(a15*a86);
  a91=(a91+a92);
  a90=(a90-a91);
  a91=(a90/a2);
  a92=(a0/a2);
  a27=(a27+a27);
  a93=(a11*a81);
  a94=(a13*a89);
  a93=(a93+a94);
  a94=(a18*a86);
  a95=(a15*a87);
  a94=(a94-a95);
  a93=(a93+a94);
  a94=(a27*a93);
  a32=(a32+a32);
  a95=(a18*a81);
  a96=(a13*a87);
  a95=(a95+a96);
  a96=(a15*a89);
  a97=(a11*a86);
  a96=(a96-a97);
  a95=(a95+a96);
  a96=(a32*a95);
  a94=(a94+a96);
  a33=(a33+a33);
  a85=(a33*a85);
  a94=(a94+a85);
  a85=(a35+a35);
  a96=(a85*a90);
  a96=(a94+a96);
  a97=(a92*a96);
  a91=(a91-a97);
  a91=(a24*a91);
  a97=casadi_sq(a23);
  a17=(a17+a97);
  a97=(a23/a17);
  a12=(a12+a12);
  a98=(1./a6);
  a10=(a10/a6);
  a5=(a5+a5);
  a99=(a10*a5);
  a98=(a98-a99);
  a99=(a11*a98);
  a14=(a14/a6);
  a100=(a14*a5);
  a101=(a13*a100);
  a99=(a99+a101);
  a19=(a19/a6);
  a101=(a19*a5);
  a102=(a18*a101);
  a16=(a16/a6);
  a103=(a16*a5);
  a104=(a15*a103);
  a102=(a102-a104);
  a99=(a99+a102);
  a102=(a12*a99);
  a20=(a20+a20);
  a104=(a18*a98);
  a105=(a13*a103);
  a104=(a104+a105);
  a105=(a15*a100);
  a106=(a11*a101);
  a105=(a105-a106);
  a104=(a104+a105);
  a105=(a20*a104);
  a102=(a102+a105);
  a21=(a21+a21);
  a105=(a15*a98);
  a106=(a13*a101);
  a105=(a105+a106);
  a106=(a11*a103);
  a107=(a18*a100);
  a106=(a106-a107);
  a105=(a105+a106);
  a106=(a21*a105);
  a102=(a102+a106);
  a106=(a22+a22);
  a107=(a102/a106);
  a108=(a97*a107);
  a22=(a22/a17);
  a17=(a13*a98);
  a109=(a11*a100);
  a110=(a18*a103);
  a109=(a109+a110);
  a110=(a15*a101);
  a109=(a109+a110);
  a17=(a17-a109);
  a109=(a22*a17);
  a108=(a108-a109);
  a108=(a4*a108);
  a108=(a4*a108);
  a109=(a0*a108);
  a91=(a91+a109);
  a91=(a91/a37);
  a109=(a46/a37);
  a110=(a34+a34);
  a94=(a94/a110);
  a111=(a94+a94);
  a111=(a109*a111);
  a91=(a91-a111);
  a111=(a3*a91);
  a88=(a88+a111);
  a111=(a15*a88);
  a112=(a94/a2);
  a113=(a43/a2);
  a96=(a113*a96);
  a112=(a112-a96);
  a112=(a24*a112);
  a108=(a43*a108);
  a112=(a112+a108);
  a108=(a13*a112);
  a111=(a111-a108);
  a95=(a95+a95);
  a95=(a46*a95);
  a108=(a41*a91);
  a95=(a95+a108);
  a108=(a18*a95);
  a111=(a111+a108);
  a93=(a93+a93);
  a93=(a46*a93);
  a91=(a38*a91);
  a93=(a93+a91);
  a91=(a11*a93);
  a111=(a111+a91);
  a91=(a111/a25);
  a40=(a40/a25);
  a108=(a40*a84);
  a91=(a91-a108);
  a108=(a4*a50);
  a96=(a11*a95);
  a114=(a15*a112);
  a115=(a13*a88);
  a114=(a114+a115);
  a96=(a96-a114);
  a114=(a18*a93);
  a96=(a96-a114);
  a96=(a48*a96);
  a86=(a86/a25);
  a114=(a48/a25);
  a115=(a114*a84);
  a86=(a86+a115);
  a86=(a49*a86);
  a96=(a96-a86);
  a86=(a15*a93);
  a115=(a18*a112);
  a116=(a11*a88);
  a115=(a115+a116);
  a116=(a13*a95);
  a115=(a115+a116);
  a86=(a86-a115);
  a86=(a51*a86);
  a87=(a87/a25);
  a115=(a51/a25);
  a116=(a115*a84);
  a87=(a87+a116);
  a87=(a52*a87);
  a86=(a86-a87);
  a96=(a96+a86);
  a88=(a18*a88);
  a112=(a11*a112);
  a88=(a88-a112);
  a95=(a15*a95);
  a88=(a88-a95);
  a93=(a13*a93);
  a88=(a88-a93);
  a88=(a53*a88);
  a89=(a89/a25);
  a93=(a53/a25);
  a95=(a93*a84);
  a89=(a89+a95);
  a89=(a45*a89);
  a88=(a88-a89);
  a96=(a96+a88);
  a81=(a81/a25);
  a88=(a1/a25);
  a84=(a88*a84);
  a81=(a81-a84);
  a81=(a44*a81);
  a111=(a1*a111);
  a81=(a81+a111);
  a96=(a96+a81);
  a96=(a47*a96);
  a108=(a108+a96);
  a91=(a91-a108);
  a105=(a105+a105);
  a105=(a56*a105);
  a108=(a17/a54);
  a96=(a55/a54);
  a23=(a23+a23);
  a17=(a23*a17);
  a102=(a102+a17);
  a17=(a96*a102);
  a108=(a108-a17);
  a108=(a36*a108);
  a17=casadi_sq(a35);
  a30=(a30+a17);
  a35=(a35/a30);
  a94=(a35*a94);
  a34=(a34/a30);
  a90=(a34*a90);
  a94=(a94-a90);
  a94=(a4*a94);
  a94=(a4*a94);
  a90=(a55*a94);
  a108=(a108+a90);
  a108=(a108/a57);
  a90=(a56/a57);
  a30=(a107+a107);
  a30=(a90*a30);
  a108=(a108-a30);
  a30=(a42*a108);
  a105=(a105+a30);
  a30=(a15*a105);
  a107=(a107/a54);
  a17=(a60/a54);
  a102=(a17*a102);
  a107=(a107-a102);
  a107=(a36*a107);
  a94=(a60*a94);
  a107=(a107+a94);
  a94=(a13*a107);
  a30=(a30-a94);
  a104=(a104+a104);
  a104=(a56*a104);
  a94=(a62*a108);
  a104=(a104+a94);
  a94=(a18*a104);
  a30=(a30+a94);
  a99=(a99+a99);
  a99=(a56*a99);
  a108=(a64*a108);
  a99=(a99+a108);
  a108=(a11*a99);
  a30=(a30+a108);
  a108=(a30/a6);
  a66=(a66/a6);
  a94=(a66*a5);
  a108=(a108-a94);
  a91=(a91+a108);
  a108=(a4*a70);
  a94=(a11*a104);
  a102=(a15*a107);
  a81=(a13*a105);
  a102=(a102+a81);
  a94=(a94-a102);
  a102=(a18*a99);
  a94=(a94-a102);
  a94=(a68*a94);
  a101=(a101/a6);
  a102=(a68/a6);
  a81=(a102*a5);
  a101=(a101+a81);
  a101=(a69*a101);
  a94=(a94-a101);
  a101=(a15*a99);
  a81=(a18*a107);
  a111=(a11*a105);
  a81=(a81+a111);
  a111=(a13*a104);
  a81=(a81+a111);
  a101=(a101-a81);
  a101=(a71*a101);
  a103=(a103/a6);
  a81=(a71/a6);
  a111=(a81*a5);
  a103=(a103+a111);
  a103=(a72*a103);
  a101=(a101-a103);
  a94=(a94+a101);
  a105=(a18*a105);
  a107=(a11*a107);
  a105=(a105-a107);
  a104=(a15*a104);
  a105=(a105-a104);
  a99=(a13*a99);
  a105=(a105-a99);
  a105=(a73*a105);
  a100=(a100/a6);
  a99=(a73/a6);
  a104=(a99*a5);
  a100=(a100+a104);
  a100=(a58*a100);
  a105=(a105-a100);
  a94=(a94+a105);
  a98=(a98/a6);
  a105=(a65/a6);
  a5=(a105*a5);
  a98=(a98-a5);
  a98=(a59*a98);
  a30=(a65*a30);
  a98=(a98+a30);
  a94=(a94+a98);
  a94=(a67*a94);
  a108=(a108+a94);
  a91=(a91-a108);
  if (res[2]!=0) res[2][7]=a91;
  a91=(a7+a7);
  a108=(a31*a91);
  a94=(a13*a108);
  a98=(a26*a91);
  a30=(a15*a98);
  a94=(a94-a30);
  a30=(1./a25);
  a5=(a28*a91);
  a30=(a30-a5);
  a5=(a18*a30);
  a100=(a29*a91);
  a104=(a11*a100);
  a5=(a5+a104);
  a94=(a94+a5);
  a5=(a94+a94);
  a5=(a46*a5);
  a104=(a11*a30);
  a107=(a18*a100);
  a104=(a104-a107);
  a107=(a15*a108);
  a104=(a104-a107);
  a107=(a13*a98);
  a104=(a104-a107);
  a107=(a104/a2);
  a101=(a18*a108);
  a103=(a15*a100);
  a101=(a101-a103);
  a103=(a11*a98);
  a111=(a13*a30);
  a103=(a103+a111);
  a101=(a101-a103);
  a103=(a27*a101);
  a111=(a13*a100);
  a84=(a18*a98);
  a111=(a111-a84);
  a84=(a11*a108);
  a89=(a15*a30);
  a84=(a84+a89);
  a111=(a111-a84);
  a84=(a32*a111);
  a103=(a103+a84);
  a94=(a33*a94);
  a103=(a103+a94);
  a94=(a85*a104);
  a94=(a103+a94);
  a84=(a92*a94);
  a107=(a107-a84);
  a107=(a24*a107);
  a7=(a7+a7);
  a84=(a19*a7);
  a89=(a18*a84);
  a95=(a16*a7);
  a112=(a15*a95);
  a89=(a89-a112);
  a112=(a10*a7);
  a86=(a11*a112);
  a87=(1./a6);
  a116=(a14*a7);
  a87=(a87-a116);
  a116=(a13*a87);
  a86=(a86+a116);
  a89=(a89-a86);
  a86=(a12*a89);
  a116=(a13*a95);
  a117=(a18*a112);
  a116=(a116-a117);
  a117=(a11*a84);
  a118=(a15*a87);
  a117=(a117+a118);
  a116=(a116-a117);
  a117=(a20*a116);
  a86=(a86+a117);
  a117=(a13*a84);
  a118=(a15*a112);
  a117=(a117-a118);
  a118=(a18*a87);
  a119=(a11*a95);
  a118=(a118+a119);
  a117=(a117+a118);
  a118=(a21*a117);
  a86=(a86+a118);
  a118=(a86/a106);
  a119=(a97*a118);
  a120=(a11*a87);
  a121=(a18*a95);
  a120=(a120-a121);
  a121=(a15*a84);
  a120=(a120-a121);
  a121=(a13*a112);
  a120=(a120-a121);
  a121=(a22*a120);
  a119=(a119-a121);
  a119=(a4*a119);
  a119=(a4*a119);
  a121=(a0*a119);
  a107=(a107+a121);
  a107=(a107/a37);
  a103=(a103/a110);
  a121=(a103+a103);
  a121=(a109*a121);
  a107=(a107-a121);
  a121=(a3*a107);
  a5=(a5+a121);
  a121=(a15*a5);
  a122=(a103/a2);
  a94=(a113*a94);
  a122=(a122-a94);
  a122=(a24*a122);
  a119=(a43*a119);
  a122=(a122+a119);
  a119=(a13*a122);
  a121=(a121-a119);
  a111=(a111+a111);
  a111=(a46*a111);
  a119=(a41*a107);
  a111=(a111+a119);
  a119=(a18*a111);
  a121=(a121+a119);
  a101=(a101+a101);
  a101=(a46*a101);
  a107=(a38*a107);
  a101=(a101+a107);
  a107=(a11*a101);
  a121=(a121+a107);
  a107=(a121/a25);
  a119=(a40*a91);
  a107=(a107-a119);
  a119=(a11*a111);
  a94=(a15*a122);
  a123=(a13*a5);
  a94=(a94+a123);
  a119=(a119-a94);
  a94=(a18*a101);
  a119=(a119-a94);
  a119=(a48*a119);
  a108=(a108/a25);
  a94=(a114*a91);
  a108=(a108+a94);
  a108=(a49*a108);
  a119=(a119-a108);
  a108=(a15*a101);
  a94=(a18*a122);
  a123=(a11*a5);
  a94=(a94+a123);
  a123=(a13*a111);
  a94=(a94+a123);
  a108=(a108-a94);
  a108=(a51*a108);
  a100=(a100/a25);
  a94=(a115*a91);
  a100=(a100+a94);
  a100=(a52*a100);
  a108=(a108-a100);
  a119=(a119+a108);
  a30=(a30/a25);
  a108=(a93*a91);
  a30=(a30-a108);
  a30=(a45*a30);
  a5=(a18*a5);
  a122=(a11*a122);
  a5=(a5-a122);
  a111=(a15*a111);
  a5=(a5-a111);
  a101=(a13*a101);
  a5=(a5-a101);
  a101=(a53*a5);
  a30=(a30+a101);
  a119=(a119+a30);
  a121=(a1*a121);
  a98=(a98/a25);
  a30=(a88*a91);
  a98=(a98+a30);
  a98=(a44*a98);
  a121=(a121-a98);
  a119=(a119+a121);
  a121=(a47*a119);
  a107=(a107-a121);
  a117=(a117+a117);
  a117=(a56*a117);
  a121=(a120/a54);
  a120=(a23*a120);
  a86=(a86+a120);
  a120=(a96*a86);
  a121=(a121-a120);
  a121=(a36*a121);
  a103=(a35*a103);
  a104=(a34*a104);
  a103=(a103-a104);
  a103=(a4*a103);
  a103=(a4*a103);
  a104=(a55*a103);
  a121=(a121+a104);
  a121=(a121/a57);
  a104=(a118+a118);
  a104=(a90*a104);
  a121=(a121-a104);
  a104=(a42*a121);
  a117=(a117+a104);
  a104=(a15*a117);
  a118=(a118/a54);
  a86=(a17*a86);
  a118=(a118-a86);
  a118=(a36*a118);
  a103=(a60*a103);
  a118=(a118+a103);
  a103=(a13*a118);
  a104=(a104-a103);
  a116=(a116+a116);
  a116=(a56*a116);
  a103=(a62*a121);
  a116=(a116+a103);
  a103=(a18*a116);
  a104=(a104+a103);
  a89=(a89+a89);
  a89=(a56*a89);
  a121=(a64*a121);
  a89=(a89+a121);
  a121=(a11*a89);
  a104=(a104+a121);
  a121=(a104/a6);
  a103=(a66*a7);
  a121=(a121-a103);
  a107=(a107+a121);
  a121=(a11*a116);
  a103=(a15*a118);
  a86=(a13*a117);
  a103=(a103+a86);
  a121=(a121-a103);
  a103=(a18*a89);
  a121=(a121-a103);
  a121=(a68*a121);
  a84=(a84/a6);
  a103=(a102*a7);
  a84=(a84+a103);
  a84=(a69*a84);
  a121=(a121-a84);
  a84=(a15*a89);
  a103=(a18*a118);
  a86=(a11*a117);
  a103=(a103+a86);
  a86=(a13*a116);
  a103=(a103+a86);
  a84=(a84-a103);
  a84=(a71*a84);
  a95=(a95/a6);
  a103=(a81*a7);
  a95=(a95+a103);
  a95=(a72*a95);
  a84=(a84-a95);
  a121=(a121+a84);
  a87=(a87/a6);
  a84=(a99*a7);
  a87=(a87-a84);
  a87=(a58*a87);
  a117=(a18*a117);
  a118=(a11*a118);
  a117=(a117-a118);
  a116=(a15*a116);
  a117=(a117-a116);
  a89=(a13*a89);
  a117=(a117-a89);
  a89=(a73*a117);
  a87=(a87+a89);
  a121=(a121+a87);
  a104=(a65*a104);
  a112=(a112/a6);
  a87=(a105*a7);
  a112=(a112+a87);
  a112=(a59*a112);
  a104=(a104-a112);
  a121=(a121+a104);
  a104=(a67*a121);
  a107=(a107-a104);
  if (res[2]!=0) res[2][8]=a107;
  a104=(a8+a8);
  a112=(a31*a104);
  a87=(a13*a112);
  a89=(a26*a104);
  a116=(a15*a89);
  a87=(a87-a116);
  a116=(a28*a104);
  a118=(a18*a116);
  a84=(1./a25);
  a95=(a29*a104);
  a84=(a84-a95);
  a95=(a11*a84);
  a118=(a118+a95);
  a87=(a87-a118);
  a118=(a87+a87);
  a118=(a46*a118);
  a95=(a18*a84);
  a103=(a11*a116);
  a95=(a95-a103);
  a103=(a15*a112);
  a95=(a95-a103);
  a103=(a13*a89);
  a95=(a95-a103);
  a103=(a95/a2);
  a86=(a13*a116);
  a120=(a11*a89);
  a86=(a86-a120);
  a120=(a15*a84);
  a98=(a18*a112);
  a120=(a120+a98);
  a86=(a86+a120);
  a120=(a27*a86);
  a98=(a15*a116);
  a30=(a11*a112);
  a98=(a98-a30);
  a30=(a18*a89);
  a101=(a13*a84);
  a30=(a30+a101);
  a98=(a98-a30);
  a30=(a32*a98);
  a120=(a120+a30);
  a87=(a33*a87);
  a120=(a120+a87);
  a87=(a85*a95);
  a87=(a120+a87);
  a30=(a92*a87);
  a103=(a103-a30);
  a103=(a24*a103);
  a8=(a8+a8);
  a30=(a14*a8);
  a101=(a13*a30);
  a111=(a10*a8);
  a122=(a11*a111);
  a101=(a101-a122);
  a122=(1./a6);
  a108=(a16*a8);
  a122=(a122-a108);
  a108=(a15*a122);
  a100=(a19*a8);
  a94=(a18*a100);
  a108=(a108+a94);
  a101=(a101+a108);
  a108=(a12*a101);
  a94=(a15*a30);
  a123=(a11*a100);
  a94=(a94-a123);
  a123=(a18*a111);
  a124=(a13*a122);
  a123=(a123+a124);
  a94=(a94-a123);
  a123=(a20*a94);
  a108=(a108+a123);
  a123=(a13*a100);
  a124=(a15*a111);
  a123=(a123-a124);
  a124=(a18*a30);
  a125=(a11*a122);
  a124=(a124+a125);
  a123=(a123-a124);
  a124=(a21*a123);
  a108=(a108+a124);
  a124=(a108/a106);
  a125=(a97*a124);
  a126=(a18*a122);
  a127=(a11*a30);
  a126=(a126-a127);
  a127=(a15*a100);
  a126=(a126-a127);
  a127=(a13*a111);
  a126=(a126-a127);
  a127=(a22*a126);
  a125=(a125-a127);
  a125=(a4*a125);
  a125=(a4*a125);
  a127=(a0*a125);
  a103=(a103+a127);
  a103=(a103/a37);
  a120=(a120/a110);
  a127=(a120+a120);
  a127=(a109*a127);
  a103=(a103-a127);
  a127=(a3*a103);
  a118=(a118+a127);
  a127=(a15*a118);
  a128=(a120/a2);
  a87=(a113*a87);
  a128=(a128-a87);
  a128=(a24*a128);
  a125=(a43*a125);
  a128=(a128+a125);
  a125=(a13*a128);
  a127=(a127-a125);
  a98=(a98+a98);
  a98=(a46*a98);
  a125=(a41*a103);
  a98=(a98+a125);
  a125=(a18*a98);
  a127=(a127+a125);
  a86=(a86+a86);
  a86=(a46*a86);
  a103=(a38*a103);
  a86=(a86+a103);
  a103=(a11*a86);
  a127=(a127+a103);
  a103=(a127/a25);
  a125=(a40*a104);
  a103=(a103-a125);
  a125=(a11*a98);
  a87=(a15*a128);
  a129=(a13*a118);
  a87=(a87+a129);
  a125=(a125-a87);
  a87=(a18*a86);
  a125=(a125-a87);
  a125=(a48*a125);
  a112=(a112/a25);
  a87=(a114*a104);
  a112=(a112+a87);
  a112=(a49*a112);
  a125=(a125-a112);
  a84=(a84/a25);
  a112=(a115*a104);
  a84=(a84-a112);
  a84=(a52*a84);
  a112=(a15*a86);
  a87=(a18*a128);
  a129=(a11*a118);
  a87=(a87+a129);
  a129=(a13*a98);
  a87=(a87+a129);
  a112=(a112-a87);
  a87=(a51*a112);
  a84=(a84+a87);
  a125=(a125+a84);
  a118=(a18*a118);
  a128=(a11*a128);
  a118=(a118-a128);
  a98=(a15*a98);
  a118=(a118-a98);
  a86=(a13*a86);
  a118=(a118-a86);
  a86=(a53*a118);
  a116=(a116/a25);
  a98=(a93*a104);
  a116=(a116+a98);
  a116=(a45*a116);
  a86=(a86-a116);
  a125=(a125+a86);
  a127=(a1*a127);
  a89=(a89/a25);
  a86=(a88*a104);
  a89=(a89+a86);
  a89=(a44*a89);
  a127=(a127-a89);
  a125=(a125+a127);
  a127=(a47*a125);
  a103=(a103-a127);
  a123=(a123+a123);
  a123=(a56*a123);
  a127=(a126/a54);
  a126=(a23*a126);
  a108=(a108+a126);
  a126=(a96*a108);
  a127=(a127-a126);
  a127=(a36*a127);
  a120=(a35*a120);
  a95=(a34*a95);
  a120=(a120-a95);
  a120=(a4*a120);
  a120=(a4*a120);
  a95=(a55*a120);
  a127=(a127+a95);
  a127=(a127/a57);
  a95=(a124+a124);
  a95=(a90*a95);
  a127=(a127-a95);
  a95=(a42*a127);
  a123=(a123+a95);
  a95=(a15*a123);
  a124=(a124/a54);
  a108=(a17*a108);
  a124=(a124-a108);
  a124=(a36*a124);
  a120=(a60*a120);
  a124=(a124+a120);
  a120=(a13*a124);
  a95=(a95-a120);
  a94=(a94+a94);
  a94=(a56*a94);
  a120=(a62*a127);
  a94=(a94+a120);
  a120=(a18*a94);
  a95=(a95+a120);
  a101=(a101+a101);
  a101=(a56*a101);
  a127=(a64*a127);
  a101=(a101+a127);
  a127=(a11*a101);
  a95=(a95+a127);
  a127=(a95/a6);
  a120=(a66*a8);
  a127=(a127-a120);
  a103=(a103+a127);
  a127=(a11*a94);
  a120=(a15*a124);
  a108=(a13*a123);
  a120=(a120+a108);
  a127=(a127-a120);
  a120=(a18*a101);
  a127=(a127-a120);
  a127=(a68*a127);
  a100=(a100/a6);
  a120=(a102*a8);
  a100=(a100+a120);
  a100=(a69*a100);
  a127=(a127-a100);
  a122=(a122/a6);
  a100=(a81*a8);
  a122=(a122-a100);
  a122=(a72*a122);
  a100=(a15*a101);
  a120=(a18*a124);
  a108=(a11*a123);
  a120=(a120+a108);
  a108=(a13*a94);
  a120=(a120+a108);
  a100=(a100-a120);
  a120=(a71*a100);
  a122=(a122+a120);
  a127=(a127+a122);
  a123=(a18*a123);
  a124=(a11*a124);
  a123=(a123-a124);
  a94=(a15*a94);
  a123=(a123-a94);
  a101=(a13*a101);
  a123=(a123-a101);
  a101=(a73*a123);
  a30=(a30/a6);
  a94=(a99*a8);
  a30=(a30+a94);
  a30=(a58*a30);
  a101=(a101-a30);
  a127=(a127+a101);
  a95=(a65*a95);
  a111=(a111/a6);
  a101=(a105*a8);
  a111=(a111+a101);
  a111=(a59*a111);
  a95=(a95-a111);
  a127=(a127+a95);
  a95=(a67*a127);
  a103=(a103-a95);
  if (res[2]!=0) res[2][9]=a103;
  a95=(a9+a9);
  a29=(a29*a95);
  a111=(a11*a29);
  a28=(a28*a95);
  a101=(a18*a28);
  a111=(a111-a101);
  a26=(a26*a95);
  a101=(a15*a26);
  a30=(1./a25);
  a31=(a31*a95);
  a30=(a30-a31);
  a31=(a13*a30);
  a101=(a101+a31);
  a111=(a111-a101);
  a101=(a111+a111);
  a101=(a46*a101);
  a31=(a15*a30);
  a94=(a11*a28);
  a124=(a18*a29);
  a94=(a94+a124);
  a31=(a31-a94);
  a94=(a13*a26);
  a31=(a31-a94);
  a94=(a31/a2);
  a124=(a13*a28);
  a122=(a11*a26);
  a124=(a124-a122);
  a122=(a15*a29);
  a120=(a18*a30);
  a122=(a122+a120);
  a124=(a124-a122);
  a27=(a27*a124);
  a122=(a13*a29);
  a120=(a18*a26);
  a122=(a122-a120);
  a120=(a11*a30);
  a108=(a15*a28);
  a120=(a120+a108);
  a122=(a122+a120);
  a32=(a32*a122);
  a27=(a27+a32);
  a33=(a33*a111);
  a27=(a27+a33);
  a85=(a85*a31);
  a85=(a27+a85);
  a92=(a92*a85);
  a94=(a94-a92);
  a94=(a24*a94);
  a9=(a9+a9);
  a14=(a14*a9);
  a92=(a13*a14);
  a10=(a10*a9);
  a33=(a11*a10);
  a92=(a92-a33);
  a16=(a16*a9);
  a33=(a15*a16);
  a111=(1./a6);
  a19=(a19*a9);
  a111=(a111-a19);
  a19=(a18*a111);
  a33=(a33+a19);
  a92=(a92-a33);
  a12=(a12*a92);
  a33=(a13*a16);
  a19=(a18*a10);
  a33=(a33-a19);
  a19=(a11*a111);
  a32=(a15*a14);
  a19=(a19+a32);
  a33=(a33+a19);
  a20=(a20*a33);
  a12=(a12+a20);
  a20=(a11*a16);
  a19=(a18*a14);
  a20=(a20-a19);
  a19=(a15*a10);
  a32=(a13*a111);
  a19=(a19+a32);
  a20=(a20-a19);
  a21=(a21*a20);
  a12=(a12+a21);
  a106=(a12/a106);
  a97=(a97*a106);
  a21=(a15*a111);
  a19=(a11*a14);
  a32=(a18*a16);
  a19=(a19+a32);
  a21=(a21-a19);
  a19=(a13*a10);
  a21=(a21-a19);
  a22=(a22*a21);
  a97=(a97-a22);
  a97=(a4*a97);
  a97=(a4*a97);
  a0=(a0*a97);
  a94=(a94+a0);
  a94=(a94/a37);
  a27=(a27/a110);
  a110=(a27+a27);
  a109=(a109*a110);
  a94=(a94-a109);
  a3=(a3*a94);
  a101=(a101+a3);
  a3=(a15*a101);
  a2=(a27/a2);
  a113=(a113*a85);
  a2=(a2-a113);
  a24=(a24*a2);
  a43=(a43*a97);
  a24=(a24+a43);
  a43=(a13*a24);
  a3=(a3-a43);
  a122=(a122+a122);
  a122=(a46*a122);
  a41=(a41*a94);
  a122=(a122+a41);
  a41=(a18*a122);
  a3=(a3+a41);
  a124=(a124+a124);
  a46=(a46*a124);
  a38=(a38*a94);
  a46=(a46+a38);
  a38=(a11*a46);
  a3=(a3+a38);
  a38=(a3/a25);
  a40=(a40*a95);
  a38=(a38-a40);
  a30=(a30/a25);
  a114=(a114*a95);
  a30=(a30-a114);
  a49=(a49*a30);
  a30=(a11*a122);
  a114=(a15*a24);
  a40=(a13*a101);
  a114=(a114+a40);
  a30=(a30-a114);
  a114=(a18*a46);
  a30=(a30-a114);
  a48=(a48*a30);
  a49=(a49+a48);
  a48=(a15*a46);
  a114=(a18*a24);
  a40=(a11*a101);
  a114=(a114+a40);
  a40=(a13*a122);
  a114=(a114+a40);
  a48=(a48-a114);
  a51=(a51*a48);
  a29=(a29/a25);
  a115=(a115*a95);
  a29=(a29+a115);
  a52=(a52*a29);
  a51=(a51-a52);
  a49=(a49+a51);
  a101=(a18*a101);
  a24=(a11*a24);
  a101=(a101-a24);
  a122=(a15*a122);
  a101=(a101-a122);
  a46=(a13*a46);
  a101=(a101-a46);
  a53=(a53*a101);
  a28=(a28/a25);
  a93=(a93*a95);
  a28=(a28+a93);
  a45=(a45*a28);
  a53=(a53-a45);
  a49=(a49+a53);
  a1=(a1*a3);
  a26=(a26/a25);
  a88=(a88*a95);
  a26=(a26+a88);
  a44=(a44*a26);
  a1=(a1-a44);
  a49=(a49+a1);
  a47=(a47*a49);
  a38=(a38-a47);
  a20=(a20+a20);
  a20=(a56*a20);
  a47=(a21/a54);
  a23=(a23*a21);
  a12=(a12+a23);
  a96=(a96*a12);
  a47=(a47-a96);
  a47=(a36*a47);
  a35=(a35*a27);
  a34=(a34*a31);
  a35=(a35-a34);
  a35=(a4*a35);
  a35=(a4*a35);
  a55=(a55*a35);
  a47=(a47+a55);
  a47=(a47/a57);
  a57=(a106+a106);
  a90=(a90*a57);
  a47=(a47-a90);
  a42=(a42*a47);
  a20=(a20+a42);
  a42=(a15*a20);
  a106=(a106/a54);
  a17=(a17*a12);
  a106=(a106-a17);
  a36=(a36*a106);
  a60=(a60*a35);
  a36=(a36+a60);
  a60=(a13*a36);
  a42=(a42-a60);
  a33=(a33+a33);
  a33=(a56*a33);
  a62=(a62*a47);
  a33=(a33+a62);
  a62=(a18*a33);
  a42=(a42+a62);
  a92=(a92+a92);
  a56=(a56*a92);
  a64=(a64*a47);
  a56=(a56+a64);
  a64=(a11*a56);
  a42=(a42+a64);
  a64=(a42/a6);
  a66=(a66*a9);
  a64=(a64-a66);
  a38=(a38+a64);
  a111=(a111/a6);
  a102=(a102*a9);
  a111=(a111-a102);
  a69=(a69*a111);
  a111=(a11*a33);
  a102=(a15*a36);
  a64=(a13*a20);
  a102=(a102+a64);
  a111=(a111-a102);
  a102=(a18*a56);
  a111=(a111-a102);
  a68=(a68*a111);
  a69=(a69+a68);
  a68=(a15*a56);
  a102=(a18*a36);
  a64=(a11*a20);
  a102=(a102+a64);
  a64=(a13*a33);
  a102=(a102+a64);
  a68=(a68-a102);
  a71=(a71*a68);
  a16=(a16/a6);
  a81=(a81*a9);
  a16=(a16+a81);
  a72=(a72*a16);
  a71=(a71-a72);
  a69=(a69+a71);
  a18=(a18*a20);
  a11=(a11*a36);
  a18=(a18-a11);
  a15=(a15*a33);
  a18=(a18-a15);
  a13=(a13*a56);
  a18=(a18-a13);
  a73=(a73*a18);
  a14=(a14/a6);
  a99=(a99*a9);
  a14=(a14+a99);
  a58=(a58*a14);
  a73=(a73-a58);
  a69=(a69+a73);
  a65=(a65*a42);
  a10=(a10/a6);
  a105=(a105*a9);
  a10=(a10+a105);
  a59=(a59*a10);
  a65=(a65-a59);
  a69=(a69+a65);
  a67=(a67*a69);
  a38=(a38-a67);
  if (res[2]!=0) res[2][10]=a38;
  if (res[2]!=0) res[2][11]=a107;
  a5=(a5/a25);
  a39=(a39/a25);
  a91=(a39*a91);
  a5=(a5-a91);
  a91=(a4*a50);
  a119=(a63*a119);
  a91=(a91+a119);
  a5=(a5-a91);
  a117=(a117/a6);
  a74=(a74/a6);
  a7=(a74*a7);
  a117=(a117-a7);
  a5=(a5+a117);
  a117=(a4*a70);
  a121=(a75*a121);
  a117=(a117+a121);
  a5=(a5-a117);
  if (res[2]!=0) res[2][12]=a5;
  a118=(a118/a25);
  a5=(a39*a104);
  a118=(a118-a5);
  a5=(a63*a125);
  a118=(a118-a5);
  a123=(a123/a6);
  a5=(a74*a8);
  a123=(a123-a5);
  a118=(a118+a123);
  a123=(a75*a127);
  a118=(a118-a123);
  if (res[2]!=0) res[2][13]=a118;
  a101=(a101/a25);
  a39=(a39*a95);
  a101=(a101-a39);
  a63=(a63*a49);
  a101=(a101-a63);
  a18=(a18/a6);
  a74=(a74*a9);
  a18=(a18-a74);
  a101=(a101+a18);
  a75=(a75*a69);
  a101=(a101-a75);
  if (res[2]!=0) res[2][14]=a101;
  if (res[2]!=0) res[2][15]=a103;
  if (res[2]!=0) res[2][16]=a118;
  a112=(a112/a25);
  a61=(a61/a25);
  a104=(a61*a104);
  a112=(a112-a104);
  a104=(a4*a50);
  a125=(a76*a125);
  a104=(a104+a125);
  a112=(a112-a104);
  a100=(a100/a6);
  a78=(a78/a6);
  a8=(a78*a8);
  a100=(a100-a8);
  a112=(a112+a100);
  a100=(a4*a70);
  a127=(a79*a127);
  a100=(a100+a127);
  a112=(a112-a100);
  if (res[2]!=0) res[2][17]=a112;
  a48=(a48/a25);
  a61=(a61*a95);
  a48=(a48-a61);
  a76=(a76*a49);
  a48=(a48-a76);
  a68=(a68/a6);
  a78=(a78*a9);
  a68=(a68-a78);
  a48=(a48+a68);
  a79=(a79*a69);
  a48=(a48-a79);
  if (res[2]!=0) res[2][18]=a48;
  if (res[2]!=0) res[2][19]=a38;
  if (res[2]!=0) res[2][20]=a101;
  if (res[2]!=0) res[2][21]=a48;
  a30=(a30/a25);
  a77=(a77/a25);
  a77=(a77*a95);
  a30=(a30-a77);
  a50=(a4*a50);
  a80=(a80*a49);
  a50=(a50+a80);
  a30=(a30-a50);
  a111=(a111/a6);
  a82=(a82/a6);
  a82=(a82*a9);
  a111=(a111-a82);
  a30=(a30+a111);
  a4=(a4*a70);
  a83=(a83*a69);
  a4=(a4+a83);
  a30=(a30-a4);
  if (res[2]!=0) res[2][22]=a30;
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
