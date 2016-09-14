/*
 * Preprocessor defines for unrolled kernels
 * D up to 17 are possible
 * Change Values in setChebyshevDegree.h for different degrees
 */

// DOF3 DEFINES

#define ADD(c, k ) u0 += coeff[c] * px[k*N]; u1 += coeff[NUMCOEFF + c] * px[k*N]; u2 += coeff[2 * NUMCOEFF + c] * px[k*N]

#define LOOP1(c, j) u0 = u1 = u2 = 0; ADD(c, 0); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP2(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP3(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP4(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP5(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP6(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP7(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); \
		v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP8(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); \
		v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP9(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); ADD(c+8, 8); \
		v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP10(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); ADD(c+8, 8); ADD(c+9, 9); \
		v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP11(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); ADD(c+8, 8); ADD(c+9, 9); \
		ADD(c+10, 10); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP12(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); ADD(c+8, 8); ADD(c+9, 9); \
		ADD(c+10, 10); ADD(c+11, 11); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP13(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); ADD(c+8, 8); ADD(c+9, 9); \
		ADD(c+10, 10); ADD(c+11, 11); ADD(c+12, 12); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP14(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); ADD(c+8, 8); ADD(c+9, 9); \
		ADD(c+10, 10); ADD(c+11, 11); ADD(c+12, 12); ADD(c+13, 13); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP15(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); ADD(c+8, 8); ADD(c+9, 9); \
		ADD(c+10, 10); ADD(c+11, 11); ADD(c+12, 12); ADD(c+13, 13); ADD(c+14, 14); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP16(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); ADD(c+8, 8); ADD(c+9, 9); \
		ADD(c+10, 10); ADD(c+11, 11); ADD(c+12, 12); ADD(c+13, 13); ADD(c+14, 14); ADD(c+15, 15); v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]
#define LOOP17(c, j) u0 = u1 = u2 = 0; ADD(c, 0); ADD(c+1, 1); ADD(c+2, 2); ADD(c+3, 3); ADD(c+4, 4); ADD(c+5, 5); ADD(c+6, 6); ADD(c+7, 7); ADD(c+8, 8); ADD(c+9, 9); \
		ADD(c+10, 10); ADD(c+11, 11); ADD(c+12, 12); ADD(c+13, 13); ADD(c+14, 14); ADD(c+15, 15); ADD(c+16, 16);v0 += u0 * py[j*N]; v1 += u1 * py[j*N]; v2 += u2 * py[j*N]



#define OUTER1(c, i)  v0 = v1 = v2 = 0; LOOP1(c, 0); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER2(c, i)  v0 = v1 = v2 = 0; LOOP2(c, 0); LOOP1(c+2, 1); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER3(c, i)  v0 = v1 = v2 = 0; LOOP3(c, 0); LOOP2(c+3, 1); LOOP1(c+5, 2); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER4(c, i)  v0 = v1 = v2 = 0; LOOP4(c, 0); LOOP3(c+4, 1); LOOP2(c+7, 2); LOOP1(c+9, 3); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER5(c, i)  v0 = v1 = v2 = 0; LOOP5(c, 0); LOOP4(c+5, 1); LOOP3(c+9, 2); LOOP2(c+12, 3); LOOP1(c+14, 4); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER6(c, i)  v0 = v1 = v2 = 0; LOOP6(c, 0); LOOP5(c+6, 1); LOOP4(c+11, 2); LOOP3(c+15, 3); LOOP2(c+18, 4); LOOP1(c+20, 5);\
		r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER7(c, i)  v0 = v1 = v2 = 0; LOOP7(c, 0); LOOP6(c+7, 1); LOOP5(c+13, 2); LOOP4(c+18, 3); LOOP3(c+22, 4); LOOP2(c+25, 5); LOOP1(c+27, 6); \
		r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER8(c, i)  v0 = v1 = v2 = 0; LOOP8(c, 0); LOOP7(c+8, 1); LOOP6(c+15, 2); LOOP5(c+21, 3); LOOP4(c+26, 4); LOOP3(c+30, 5); LOOP2(c+33, 6); LOOP1(c+35, 7); \
		r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER9(c, i)  v0 = v1 = v2 = 0; LOOP9(c, 0); LOOP8(c+9, 1); LOOP7(c+17, 2); LOOP6(c+24, 3); LOOP5(c+30, 4); LOOP4(c+35, 5); LOOP3(c+39, 6); LOOP2(c+42, 7); \
		LOOP1(c+44, 8); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER10(c, i)  v0 = v1 = v2 = 0; LOOP10(c, 0); LOOP9(c+10, 1); LOOP8(c+19, 2); LOOP7(c+27, 3); LOOP6(c+34, 4); LOOP5(c+40, 5); LOOP4(c+45, 6); LOOP3(c+49, 7); \
		LOOP2(c+52, 8); LOOP1(c+54, 9); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER11(c, i)  v0 = v1 = v2 = 0; LOOP11(c, 0); LOOP10(c+11, 1); LOOP9(c+21, 2); LOOP8(c+30, 3); LOOP7(c+38, 4); LOOP6(c+45, 5); LOOP5(c+51, 6); LOOP4(c+56, 7); \
		LOOP3(c+60, 8); LOOP2(c+63, 9); LOOP1(c+65, 10); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER12(c, i)  v0 = v1 = v2 = 0; LOOP12(c, 0); LOOP11(c+12, 1); LOOP10(c+23, 2); LOOP9(c+33, 3); LOOP8(c+42, 4); LOOP7(c+50, 5); LOOP6(c+57, 6); LOOP5(c+63, 7); \
		LOOP4(c+68, 8); LOOP3(c+72, 9); LOOP2(c+75, 10); LOOP1(c+77, 11); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER13(c, i)  v0 = v1 = v2 = 0; LOOP13(c, 0); LOOP12(c+13, 1); LOOP11(c+25, 2); LOOP10(c+36, 3); LOOP9(c+46, 4); LOOP8(c+55, 5); LOOP7(c+63, 6); LOOP6(c+70, 7); \
		LOOP5(c+76, 8); LOOP4(c+81, 9); LOOP3(c+85, 10); LOOP2(c+88, 11); LOOP1(c+90, 12); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER14(c, i)  v0 = v1 = v2 = 0; LOOP14(c, 0); LOOP13(c+14, 1); LOOP12(c+27, 2); LOOP11(c+39, 3); LOOP10(c+50, 4); LOOP9(c+60, 5); LOOP8(c+69, 6); LOOP7(c+77, 7); \
		LOOP6(c+84, 8); LOOP5(c+90, 9); LOOP4(c+95, 10); LOOP3(c+99, 11); LOOP2(c+102, 12); LOOP1(c+104, 13);r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER15(c, i)  v0 = v1 = v2 = 0; LOOP15(c, 0); LOOP14(c+15, 1); LOOP13(c+29, 2); LOOP12(c+42, 3); LOOP11(c+54, 4); LOOP10(c+65, 5); LOOP9(c+75, 6); LOOP8(c+84, 7); \
		LOOP7(c+92, 8); LOOP6(c+99, 9); LOOP5(c+105, 10); LOOP4(c+110, 11); LOOP3(c+114, 12); LOOP2(c+117, 13); LOOP1(c+119, 14);r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER16(c, i)  v0 = v1 = v2 = 0; LOOP16(c, 0); LOOP15(c+16, 1); LOOP14(c+31, 2); LOOP13(c+45, 3); LOOP12(c+58, 4); LOOP11(c+70, 5); LOOP10(c+81, 6); LOOP9(c+91, 7); \
		LOOP8(c+100, 8); LOOP7(c+108, 9); LOOP6(c+115, 10); LOOP5(c+121, 11); LOOP4(c+126, 12); LOOP3(c+130, 13); LOOP2(c+133, 14); \
		LOOP1(c+135 ,15); r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]
#define OUTER17(c, i)  v0 = v1 = v2 = 0; LOOP17(c, 0); LOOP16(c+17, 1); LOOP15(c+33, 2); LOOP14(c+48, 3); LOOP13(c+62, 4); LOOP12(c+75, 5); LOOP11(c+87, 6); LOOP10(c+98, 7); \
		LOOP9(c+108, 8); LOOP8(c+117, 9); LOOP7(c+125, 10); LOOP6(c+132, 11); LOOP5(c+138, 12); LOOP4(c+143, 13); LOOP3(c+147, 14); LOOP2(c+150 ,15); \
		LOOP1(c+152 ,16);r0 += v0 * pz[i*N]; r1 += v1 * pz[i*N]; r2 += v2 * pz[i*N]

#define COMP1() OUTER1(0, 0)
#define COMP2() OUTER2(0, 0); OUTER1(3, 1)
#define COMP3() OUTER3(0, 0); OUTER2(6, 1); OUTER1(9, 2)
#define COMP4() OUTER4(0, 0); OUTER3(10, 1); OUTER2(16, 2); OUTER1(19, 3)
#define COMP5() OUTER5(0, 0); OUTER4(15, 1); OUTER3(25, 2); OUTER2(31, 3); OUTER1(34, 4)
#define COMP6() OUTER6(0, 0); OUTER5(21, 1); OUTER4(36, 2); OUTER3(46, 3); OUTER2(52, 4); OUTER1(55, 5)
#define COMP7() OUTER7(0, 0); OUTER6(28, 1); OUTER5(49, 2); OUTER4(64, 3); OUTER3(74, 4); \
			   OUTER2(80, 5); OUTER1(83, 6)
#define COMP8() OUTER8(0, 0); OUTER7(36, 1); OUTER6(64, 2); OUTER5(85, 3); OUTER4(100, 4); \
			   OUTER3(110, 5); OUTER2(116, 6); OUTER1(119, 7)
#define COMP9() OUTER9(0, 0); OUTER8(45, 1); OUTER7(81, 2); OUTER6(109, 3); OUTER5(130, 4); \
			   OUTER4(145, 5); OUTER3(155, 6); OUTER2(161, 7); OUTER1(164, 8)
#define COMP10() OUTER10(0, 0); OUTER9(55, 1); OUTER8(100, 2); OUTER7(136, 3); OUTER6(164, 4); \
			   OUTER5(185, 5); OUTER4(200, 6); OUTER3(210, 7); OUTER2(216, 8); OUTER1(219, 9)
#define COMP11() OUTER11(0, 0); OUTER10(66, 1); OUTER9(121, 2); OUTER8(166, 3); OUTER7(202, 4); \
			   OUTER6(230, 5); OUTER5(251, 6); OUTER4(266, 7); OUTER3(276, 8); OUTER2(282, 9); \
			   OUTER1(285, 10)
#define COMP12() OUTER12(0, 0); OUTER11(78, 1); OUTER10(144, 2); OUTER9(199, 3); OUTER8(244, 4); \
			   OUTER7(280, 5); OUTER6(308, 6); OUTER5(329, 7); OUTER4(344, 8); OUTER3(354, 9); \
			   OUTER2(360, 10); OUTER1(363, 11)
#define COMP13() OUTER13(0, 0); OUTER12(91, 1); OUTER11(169, 2); OUTER10(235, 3); OUTER9(290, 4); \
			   OUTER8(335, 5); OUTER7(371, 6); OUTER6(399, 7); OUTER5(420, 8); OUTER4(435, 9); \
			   OUTER3(445, 10); OUTER2(451, 11); OUTER1(454, 12)
#define COMP14() OUTER14(0, 0); OUTER13(105, 1); OUTER12(196, 2); OUTER11(274, 3); OUTER10(340, 4); \
			   OUTER9(395, 5); OUTER8(440, 6); OUTER7(476, 7); OUTER6(504, 8); OUTER5(525, 9); \
			   OUTER4(540, 10); OUTER3(550, 11); OUTER2(556, 12); OUTER1(559, 13)
#define COMP15() OUTER15(0, 0); OUTER14(120, 1); OUTER13(225, 2); OUTER12(316, 3); OUTER11(394, 4); \
			   OUTER10(460, 5); OUTER9(515, 6); OUTER8(560, 7); OUTER7(596, 8); OUTER6(624, 9); \
			   OUTER5(645, 10); OUTER4(660, 11); OUTER3(670, 12); OUTER2(676, 13); OUTER1(679, 14)
#define COMP16() OUTER16(0, 0); OUTER15(136, 1); OUTER14(256, 2); OUTER13(361, 3); OUTER12(452, 4); \
			   OUTER11(530, 5); OUTER10(596, 6); OUTER9(651, 7); OUTER8(696, 8); OUTER7(732, 9); \
			   OUTER6(760, 10); OUTER5(781, 11); OUTER4(796, 12); OUTER3(806, 13); OUTER2(812, 14); \
			   OUTER1(815, 15)
#define COMP17() OUTER17(0, 0); OUTER16(153, 1); OUTER15(289, 2); OUTER14(409, 3); OUTER13(514, 4); \
			   OUTER12(605, 5); OUTER11(683, 6); OUTER10(749, 7); OUTER9(804, 8); OUTER8(849, 9); \
			   OUTER7(885, 10); OUTER6(913, 11); OUTER5(934, 12); OUTER4(949, 13); OUTER3(959, 14); \
			   OUTER2(965, 15); OUTER1(968, 16)
			   
// DOF1 DEFINES

#define ADD_1(c, k ) u0 += coeff[c] * px[k*N]

#define LOOP1_1(c, j) u0 = 0; ADD_1(c, 0); v0 += u0 * py[j*N]
#define LOOP2_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); v0 += u0 * py[j*N]
#define LOOP3_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); v0 += u0 * py[j*N]
#define LOOP4_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); v0 += u0 * py[j*N]
#define LOOP5_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); v0 += u0 * py[j*N]
#define LOOP6_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); v0 += u0 * py[j*N]
#define LOOP7_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); \
		v0 += u0 * py[j*N]
#define LOOP8_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); \
		v0 += u0 * py[j*N]
#define LOOP9_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); ADD_1(c+8, 8); \
		v0 += u0 * py[j*N]
#define LOOP10_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); ADD_1(c+8, 8); ADD_1(c+9, 9); \
		v0 += u0 * py[j*N]
#define LOOP11_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); ADD_1(c+8, 8); ADD_1(c+9, 9); \
		ADD_1(c+10, 10); v0 += u0 * py[j*N]
#define LOOP12_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); ADD_1(c+8, 8); ADD_1(c+9, 9); \
		ADD_1(c+10, 10); ADD_1(c+11, 11); v0 += u0 * py[j*N]
#define LOOP13_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); ADD_1(c+8, 8); ADD_1(c+9, 9); \
		ADD_1(c+10, 10); ADD_1(c+11, 11); ADD_1(c+12, 12); v0 += u0 * py[j*N]
#define LOOP14_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); ADD_1(c+8, 8); ADD_1(c+9, 9); \
		ADD_1(c+10, 10); ADD_1(c+11, 11); ADD_1(c+12, 12); ADD_1(c+13, 13); v0 += u0 * py[j*N]
#define LOOP15_1(c, j) u0= 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); ADD_1(c+8, 8); ADD_1(c+9, 9); \
		ADD_1(c+10, 10); ADD_1(c+11, 11); ADD_1(c+12, 12); ADD_1(c+13, 13); ADD_1(c+14, 14); v0 += u0 * py[j*N]
#define LOOP16_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); ADD_1(c+8, 8); ADD_1(c+9, 9); \
		ADD_1(c+10, 10); ADD_1(c+11, 11); ADD_1(c+12, 12); ADD_1(c+13, 13); ADD_1(c+14, 14); ADD_1(c+15, 15); v0 += u0 * py[j*N]
#define LOOP17_1(c, j) u0 = 0; ADD_1(c, 0); ADD_1(c+1, 1); ADD_1(c+2, 2); ADD_1(c+3, 3); ADD_1(c+4, 4); ADD_1(c+5, 5); ADD_1(c+6, 6); ADD_1(c+7, 7); ADD_1(c+8, 8); ADD_1(c+9, 9); \
		ADD_1(c+10, 10); ADD_1(c+11, 11); ADD_1(c+12, 12); ADD_1(c+13, 13); ADD_1(c+14, 14); ADD_1(c+15, 15); ADD_1(c+16, 16);v0 += u0 * py[j*N]

#define OUTER1_1(c, i)  v0 = 0; LOOP1_1(c, 0); r0 += v0 * pz[i*N];
#define OUTER2_1(c, i)  v0 = 0; LOOP2_1(c, 0); LOOP1_1(c+2, 1); r0 += v0 * pz[i*N];
#define OUTER3_1(c, i)  v0 = 0; LOOP3_1(c, 0); LOOP2_1(c+3, 1); LOOP1_1(c+5, 2); r0 += v0 * pz[i*N];
#define OUTER4_1(c, i)  v0 = 0; LOOP4_1(c, 0); LOOP3_1(c+4, 1); LOOP2_1(c+7, 2); LOOP1_1(c+9, 3); r0 += v0 * pz[i*N];
#define OUTER5_1(c, i)  v0 = 0; LOOP5_1(c, 0); LOOP4_1(c+5, 1); LOOP3_1(c+9, 2); LOOP2_1(c+12, 3); LOOP1_1(c+14, 4); r0 += v0 * pz[i*N];
#define OUTER6_1(c, i)  v0 = 0; LOOP6_1(c, 0); LOOP5_1(c+6, 1); LOOP4_1(c+11, 2); LOOP3_1(c+15, 3); LOOP2_1(c+18, 4); LOOP1_1(c+20, 5);\
		r0 += v0 * pz[i*N]
#define OUTER7_1(c, i)  v0 = 0; LOOP7_1(c, 0); LOOP6_1(c+7, 1); LOOP5_1(c+13, 2); LOOP4_1(c+18, 3); LOOP3_1(c+22, 4); LOOP2_1(c+25, 5); LOOP1_1(c+27, 6); \
		r0 += v0 * pz[i*N]
#define OUTER8_1(c, i)  v0 = 0; LOOP8_1(c, 0); LOOP7_1(c+8, 1); LOOP6_1(c+15, 2); LOOP5_1(c+21, 3); LOOP4_1(c+26, 4); LOOP3_1(c+30, 5); LOOP2_1(c+33, 6); LOOP1_1(c+35, 7); \
		r0 += v0 * pz[i*N]
#define OUTER9_1(c, i)  v0 = 0; LOOP9_1(c, 0); LOOP8_1(c+9, 1); LOOP7_1(c+17, 2); LOOP6_1(c+24, 3); LOOP5_1(c+30, 4); LOOP4_1(c+35, 5); LOOP3_1(c+39, 6); LOOP2_1(c+42, 7); \
		LOOP1_1(c+44, 8); r0 += v0 * pz[i*N]
#define OUTER10_1(c, i)  v0 = 0; LOOP10_1(c, 0); LOOP9_1(c+10, 1); LOOP8_1(c+19, 2); LOOP7_1(c+27, 3); LOOP6_1(c+34, 4); LOOP5_1(c+40, 5); LOOP4_1(c+45, 6); LOOP3_1(c+49, 7); \
		LOOP2_1(c+52, 8); LOOP1_1(c+54, 9); r0 += v0 * pz[i*N]
#define OUTER11_1(c, i)  v0 = 0; LOOP11_1(c, 0); LOOP10_1(c+11, 1); LOOP9_1(c+21, 2); LOOP8_1(c+30, 3); LOOP7_1(c+38, 4); LOOP6_1(c+45, 5); LOOP5_1(c+51, 6); LOOP4_1(c+56, 7); \
		LOOP3_1(c+60, 8); LOOP2_1(c+63, 9); LOOP1_1(c+65, 10); r0 += v0 * pz[i*N]
#define OUTER12_1(c, i)  v0 = 0; LOOP12_1(c, 0); LOOP11_1(c+12, 1); LOOP10_1(c+23, 2); LOOP9_1(c+33, 3); LOOP8_1(c+42, 4); LOOP7_1(c+50, 5); LOOP6_1(c+57, 6); LOOP5_1(c+63, 7); \
		LOOP4_1(c+68, 8); LOOP3_1(c+72, 9); LOOP2_1(c+75, 10); LOOP1_1(c+77, 11); r0 += v0 * pz[i*N]
#define OUTER13_1(c, i)  v0 = 0; LOOP13_1(c, 0); LOOP12_1(c+13, 1); LOOP11_1(c+25, 2); LOOP10_1(c+36, 3); LOOP9_1(c+46, 4); LOOP8_1(c+55, 5); LOOP7_1(c+63, 6); LOOP6_1(c+70, 7); \
		LOOP5_1(c+76, 8); LOOP4_1(c+81, 9); LOOP3_1(c+85, 10); LOOP2_1(c+88, 11); LOOP1_1(c+90, 12); r0 += v0 * pz[i*N]
#define OUTER14_1(c, i)  v0 = 0; LOOP14_1(c, 0); LOOP13_1(c+14, 1); LOOP12_1(c+27, 2); LOOP11_1(c+39, 3); LOOP10_1(c+50, 4); LOOP9_1(c+60, 5); LOOP8_1(c+69, 6); LOOP7_1(c+77, 7); \
		LOOP6_1(c+84, 8); LOOP5_1(c+90, 9); LOOP4_1(c+95, 10); LOOP3_1(c+99, 11); LOOP2_1(c+102, 12); LOOP1_1(c+104, 13);r0 += v0 * pz[i*N]
#define OUTER15_1(c, i)  v0 = 0; LOOP15_1(c, 0); LOOP14_1(c+15, 1); LOOP13_1(c+29, 2); LOOP12_1(c+42, 3); LOOP11_1(c+54, 4); LOOP10_1(c+65, 5); LOOP9_1(c+75, 6); LOOP8_1(c+84, 7); \
		LOOP7_1(c+92, 8); LOOP6_1(c+99, 9); LOOP5_1(c+105, 10); LOOP4_1(c+110, 11); LOOP3_1(c+114, 12); LOOP2_1(c+117, 13); LOOP1_1(c+119, 14); r0 += v0 * pz[i*N]
#define OUTER16_1(c, i)  v0 = 0; LOOP16_1(c, 0); LOOP15_1(c+16, 1); LOOP14_1(c+31, 2); LOOP13_1(c+45, 3); LOOP12_1(c+58, 4); LOOP11_1(c+70, 5); LOOP10_1(c+81, 6); LOOP9_1(c+91, 7); \
		LOOP8_1(c+100, 8); LOOP7_1(c+108, 9); LOOP6_1(c+115, 10); LOOP5_1(c+121, 11); LOOP4_1(c+126, 12); LOOP3_1(c+130, 13); LOOP2_1(c+133, 14); LOOP1_1(c+135 ,15); r0 += v0 * pz[i*N]
#define OUTER17_1(c, i)  v0 = 0; LOOP17_1(c, 0); LOOP16_1(c+17, 1); LOOP15_1(c+33, 2); LOOP14_1(c+48, 3); LOOP13_1(c+62, 4); LOOP12_1(c+75, 5); LOOP11_1(c+87, 6); LOOP10_1(c+98, 7); \
		LOOP9_1(c+108, 8); LOOP8_1(c+117, 9); LOOP7_1(c+125, 10); LOOP6_1(c+132, 11); LOOP5_1(c+138, 12); LOOP4_1(c+143, 13); LOOP3_1(c+147, 14); LOOP2_1(c+150 ,15); \
		LOOP1_1(c+152 ,16);r0 += v0 * pz[i*N]

#define COMP1_1() OUTER1_1(0, 0)
#define COMP2_1() OUTER2_1(0, 0); OUTER1_1(3, 1)
#define COMP3_1() OUTER3_1(0, 0); OUTER2_1(6, 1); OUTER1_1(9, 2)
#define COMP4_1() OUTER4_1(0, 0); OUTER3_1(10, 1); OUTER2_1(16, 2); OUTER1_1(19, 3)
#define COMP5_1() OUTER5_1(0, 0); OUTER4_1(15, 1); OUTER3_1(25, 2); OUTER2_1(31, 3); OUTER1_1(34, 4)
#define COMP6_1() OUTER6_1(0, 0); OUTER5_1(21, 1); OUTER4_1(36, 2); OUTER3_1(46, 3); OUTER2_1(52, 4); OUTER1_1(55, 5)
#define COMP7_1() OUTER7_1(0, 0); OUTER6_1(28, 1); OUTER5_1(49, 2); OUTER4_1(64, 3); OUTER3_1(74, 4); \
			   OUTER2_1(80, 5); OUTER1_1(83, 6)
#define COMP8_1() OUTER8_1(0, 0); OUTER7_1(36, 1); OUTER6_1(64, 2); OUTER5_1(85, 3); OUTER4_1(100, 4); \
			   OUTER3_1(110, 5); OUTER2_1(116, 6); OUTER1_1(119, 7)
#define COMP9_1() OUTER9_1(0, 0); OUTER8_1(45, 1); OUTER7_1(81, 2); OUTER6_1(109, 3); OUTER5_1(130, 4); \
			   OUTER4_1(145, 5); OUTER3_1(155, 6); OUTER2_1(161, 7); OUTER1_1(164, 8)
#define COMP10_1() OUTER10_1(0, 0); OUTER9_1(55, 1); OUTER8_1(100, 2); OUTER7_1(136, 3); OUTER6_1(164, 4); \
			   OUTER5_1(185, 5); OUTER4_1(200, 6); OUTER3_1(210, 7); OUTER2_1(216, 8); OUTER1_1(219, 9)
#define COMP11_1() OUTER11_1(0, 0); OUTER10_1(66, 1); OUTER9_1(121, 2); OUTER8_1(166, 3); OUTER7_1(202, 4); \
			   OUTER6_1(230, 5); OUTER5_1(251, 6); OUTER4_1(266, 7); OUTER3_1(276, 8); OUTER2_1(282, 9); \
			   OUTER1_1(285, 10)
#define COMP12_1() OUTER12_1(0, 0); OUTER11_1(78, 1); OUTER10_1(144, 2); OUTER9_1(199, 3); OUTER8_1(244, 4); \
			   OUTER7_1(280, 5); OUTER6_1(308, 6); OUTER5_1(329, 7); OUTER4_1(344, 8); OUTER3_1(354, 9); \
			   OUTER2_1(360, 10); OUTER1_1(363, 11)
#define COMP13_1() OUTER13_1(0, 0); OUTER12_1(91, 1); OUTER11_1(169, 2); OUTER10_1(235, 3); OUTER9_1(290, 4); \
			   OUTER8_1(335, 5); OUTER7_1(371, 6); OUTER6_1(399, 7); OUTER5_1(420, 8); OUTER4_1(435, 9); \
			   OUTER3_1(445, 10); OUTER2_1(451, 11); OUTER1_1(454, 12)
#define COMP14_1() OUTER14_1(0, 0); OUTER13_1(105, 1); OUTER12_1(196, 2); OUTER11_1(274, 3); OUTER10_1(340, 4); \
			   OUTER9_1(395, 5); OUTER8_1(440, 6); OUTER7_1(476, 7); OUTER6_1(504, 8); OUTER5_1(525, 9); \
			   OUTER4_1(540, 10); OUTER3_1(550, 11); OUTER2_1(556, 12); OUTER1_1(559, 13)
#define COMP15_1() OUTER15_1(0, 0); OUTER14_1(120, 1); OUTER13_1(225, 2); OUTER12_1(316, 3); OUTER11_1(394, 4); \
			   OUTER10_1(460, 5); OUTER9_1(515, 6); OUTER8_1(560, 7); OUTER7_1(596, 8); OUTER6_1(624, 9); \
			   OUTER5_1(645, 10); OUTER4_1(660, 11); OUTER3_1(670, 12); OUTER2_1(676, 13); OUTER1_1(679, 14)
#define COMP16_1() OUTER16_1(0, 0); OUTER15_1(136, 1); OUTER14_1(256, 2); OUTER13_1(361, 3); OUTER12_1(452, 4); \
			   OUTER11_1(530, 5); OUTER10_1(596, 6); OUTER9_1(651, 7); OUTER8_1(696, 8); OUTER7_1(732, 9); \
			   OUTER6_1(760, 10); OUTER5_1(781, 11); OUTER4_1(796, 12); OUTER3_1(806, 13); OUTER2_1(812, 14); \
			   OUTER1_1(815, 15)
#define COMP17_1() OUTER17_1(0, 0); OUTER16_1(153, 1); OUTER15_1(289, 2); OUTER14_1(409, 3); OUTER13_1(514, 4); \
			   OUTER12_1(605, 5); OUTER11_1(683, 6); OUTER10_1(749, 7); OUTER9_1(804, 8); OUTER8_1(849, 9); \
			   OUTER7_1(885, 10); OUTER6_1(913, 11); OUTER5_1(934, 12); OUTER4_1(949, 13); OUTER3_1(959, 14); \
			   OUTER2_1(965, 15); OUTER1_1(968, 16)	   
