`timescale 1ns/1ps

module relu32 (
    input [255:0] inp,
    output [223:0] out
);

    // verilator lint_off UNUSEDSIGNAL
    // Explicit quantization operation will drop bits if exists

    wire [7:0] v0; assign v0[7:0] = inp[7:0]; // 0.0
    wire [7:0] v1; assign v1[7:0] = inp[15:8]; // 0.0
    wire [7:0] v2; assign v2[7:0] = inp[23:16]; // 0.0
    wire [7:0] v3; assign v3[7:0] = inp[31:24]; // 0.0
    wire [7:0] v4; assign v4[7:0] = inp[39:32]; // 0.0
    wire [7:0] v5; assign v5[7:0] = inp[47:40]; // 0.0
    wire [7:0] v6; assign v6[7:0] = inp[55:48]; // 0.0
    wire [7:0] v7; assign v7[7:0] = inp[63:56]; // 0.0
    wire [7:0] v8; assign v8[7:0] = inp[71:64]; // 0.0
    wire [7:0] v9; assign v9[7:0] = inp[79:72]; // 0.0
    wire [7:0] v10; assign v10[7:0] = inp[87:80]; // 0.0
    wire [7:0] v11; assign v11[7:0] = inp[95:88]; // 0.0
    wire [7:0] v12; assign v12[7:0] = inp[103:96]; // 0.0
    wire [7:0] v13; assign v13[7:0] = inp[111:104]; // 0.0
    wire [7:0] v14; assign v14[7:0] = inp[119:112]; // 0.0
    wire [7:0] v15; assign v15[7:0] = inp[127:120]; // 0.0
    wire [7:0] v16; assign v16[7:0] = inp[135:128]; // 0.0
    wire [7:0] v17; assign v17[7:0] = inp[143:136]; // 0.0
    wire [7:0] v18; assign v18[7:0] = inp[151:144]; // 0.0
    wire [7:0] v19; assign v19[7:0] = inp[159:152]; // 0.0
    wire [7:0] v20; assign v20[7:0] = inp[167:160]; // 0.0
    wire [7:0] v21; assign v21[7:0] = inp[175:168]; // 0.0
    wire [7:0] v22; assign v22[7:0] = inp[183:176]; // 0.0
    wire [7:0] v23; assign v23[7:0] = inp[191:184]; // 0.0
    wire [7:0] v24; assign v24[7:0] = inp[199:192]; // 0.0
    wire [7:0] v25; assign v25[7:0] = inp[207:200]; // 0.0
    wire [7:0] v26; assign v26[7:0] = inp[215:208]; // 0.0
    wire [7:0] v27; assign v27[7:0] = inp[223:216]; // 0.0
    wire [7:0] v28; assign v28[7:0] = inp[231:224]; // 0.0
    wire [7:0] v29; assign v29[7:0] = inp[239:232]; // 0.0
    wire [7:0] v30; assign v30[7:0] = inp[247:240]; // 0.0
    wire [7:0] v31; assign v31[7:0] = inp[255:248]; // 0.0
    wire [7:0] v32; assign v32[7:0] = v0[7:0]; // 0.0
    wire [6:0] v33; assign v33[6:0] = v32[6:0] & {7{~v32[7]}}; // 0.0
    wire [7:0] v34; assign v34[7:0] = v1[7:0]; // 0.0
    wire [6:0] v35; assign v35[6:0] = v34[6:0] & {7{~v34[7]}}; // 0.0
    wire [7:0] v36; assign v36[7:0] = v2[7:0]; // 0.0
    wire [6:0] v37; assign v37[6:0] = v36[6:0] & {7{~v36[7]}}; // 0.0
    wire [7:0] v38; assign v38[7:0] = v3[7:0]; // 0.0
    wire [6:0] v39; assign v39[6:0] = v38[6:0] & {7{~v38[7]}}; // 0.0
    wire [7:0] v40; assign v40[7:0] = v4[7:0]; // 0.0
    wire [6:0] v41; assign v41[6:0] = v40[6:0] & {7{~v40[7]}}; // 0.0
    wire [7:0] v42; assign v42[7:0] = v5[7:0]; // 0.0
    wire [6:0] v43; assign v43[6:0] = v42[6:0] & {7{~v42[7]}}; // 0.0
    wire [7:0] v44; assign v44[7:0] = v6[7:0]; // 0.0
    wire [6:0] v45; assign v45[6:0] = v44[6:0] & {7{~v44[7]}}; // 0.0
    wire [7:0] v46; assign v46[7:0] = v7[7:0]; // 0.0
    wire [6:0] v47; assign v47[6:0] = v46[6:0] & {7{~v46[7]}}; // 0.0
    wire [7:0] v48; assign v48[7:0] = v8[7:0]; // 0.0
    wire [6:0] v49; assign v49[6:0] = v48[6:0] & {7{~v48[7]}}; // 0.0
    wire [7:0] v50; assign v50[7:0] = v9[7:0]; // 0.0
    wire [6:0] v51; assign v51[6:0] = v50[6:0] & {7{~v50[7]}}; // 0.0
    wire [7:0] v52; assign v52[7:0] = v10[7:0]; // 0.0
    wire [6:0] v53; assign v53[6:0] = v52[6:0] & {7{~v52[7]}}; // 0.0
    wire [7:0] v54; assign v54[7:0] = v11[7:0]; // 0.0
    wire [6:0] v55; assign v55[6:0] = v54[6:0] & {7{~v54[7]}}; // 0.0
    wire [7:0] v56; assign v56[7:0] = v12[7:0]; // 0.0
    wire [6:0] v57; assign v57[6:0] = v56[6:0] & {7{~v56[7]}}; // 0.0
    wire [7:0] v58; assign v58[7:0] = v13[7:0]; // 0.0
    wire [6:0] v59; assign v59[6:0] = v58[6:0] & {7{~v58[7]}}; // 0.0
    wire [7:0] v60; assign v60[7:0] = v14[7:0]; // 0.0
    wire [6:0] v61; assign v61[6:0] = v60[6:0] & {7{~v60[7]}}; // 0.0
    wire [7:0] v62; assign v62[7:0] = v15[7:0]; // 0.0
    wire [6:0] v63; assign v63[6:0] = v62[6:0] & {7{~v62[7]}}; // 0.0
    wire [7:0] v64; assign v64[7:0] = v16[7:0]; // 0.0
    wire [6:0] v65; assign v65[6:0] = v64[6:0] & {7{~v64[7]}}; // 0.0
    wire [7:0] v66; assign v66[7:0] = v17[7:0]; // 0.0
    wire [6:0] v67; assign v67[6:0] = v66[6:0] & {7{~v66[7]}}; // 0.0
    wire [7:0] v68; assign v68[7:0] = v18[7:0]; // 0.0
    wire [6:0] v69; assign v69[6:0] = v68[6:0] & {7{~v68[7]}}; // 0.0
    wire [7:0] v70; assign v70[7:0] = v19[7:0]; // 0.0
    wire [6:0] v71; assign v71[6:0] = v70[6:0] & {7{~v70[7]}}; // 0.0
    wire [7:0] v72; assign v72[7:0] = v20[7:0]; // 0.0
    wire [6:0] v73; assign v73[6:0] = v72[6:0] & {7{~v72[7]}}; // 0.0
    wire [7:0] v74; assign v74[7:0] = v21[7:0]; // 0.0
    wire [6:0] v75; assign v75[6:0] = v74[6:0] & {7{~v74[7]}}; // 0.0
    wire [7:0] v76; assign v76[7:0] = v22[7:0]; // 0.0
    wire [6:0] v77; assign v77[6:0] = v76[6:0] & {7{~v76[7]}}; // 0.0
    wire [7:0] v78; assign v78[7:0] = v23[7:0]; // 0.0
    wire [6:0] v79; assign v79[6:0] = v78[6:0] & {7{~v78[7]}}; // 0.0
    wire [7:0] v80; assign v80[7:0] = v24[7:0]; // 0.0
    wire [6:0] v81; assign v81[6:0] = v80[6:0] & {7{~v80[7]}}; // 0.0
    wire [7:0] v82; assign v82[7:0] = v25[7:0]; // 0.0
    wire [6:0] v83; assign v83[6:0] = v82[6:0] & {7{~v82[7]}}; // 0.0
    wire [7:0] v84; assign v84[7:0] = v26[7:0]; // 0.0
    wire [6:0] v85; assign v85[6:0] = v84[6:0] & {7{~v84[7]}}; // 0.0
    wire [7:0] v86; assign v86[7:0] = v27[7:0]; // 0.0
    wire [6:0] v87; assign v87[6:0] = v86[6:0] & {7{~v86[7]}}; // 0.0
    wire [7:0] v88; assign v88[7:0] = v28[7:0]; // 0.0
    wire [6:0] v89; assign v89[6:0] = v88[6:0] & {7{~v88[7]}}; // 0.0
    wire [7:0] v90; assign v90[7:0] = v29[7:0]; // 0.0
    wire [6:0] v91; assign v91[6:0] = v90[6:0] & {7{~v90[7]}}; // 0.0
    wire [7:0] v92; assign v92[7:0] = v30[7:0]; // 0.0
    wire [6:0] v93; assign v93[6:0] = v92[6:0] & {7{~v92[7]}}; // 0.0
    wire [7:0] v94; assign v94[7:0] = v31[7:0]; // 0.0
    wire [6:0] v95; assign v95[6:0] = v94[6:0] & {7{~v94[7]}}; // 0.0

    // verilator lint_on UNUSEDSIGNAL

    assign out[6:0] = v33[6:0];
    assign out[13:7] = v35[6:0];
    assign out[20:14] = v37[6:0];
    assign out[27:21] = v39[6:0];
    assign out[34:28] = v41[6:0];
    assign out[41:35] = v43[6:0];
    assign out[48:42] = v45[6:0];
    assign out[55:49] = v47[6:0];
    assign out[62:56] = v49[6:0];
    assign out[69:63] = v51[6:0];
    assign out[76:70] = v53[6:0];
    assign out[83:77] = v55[6:0];
    assign out[90:84] = v57[6:0];
    assign out[97:91] = v59[6:0];
    assign out[104:98] = v61[6:0];
    assign out[111:105] = v63[6:0];
    assign out[118:112] = v65[6:0];
    assign out[125:119] = v67[6:0];
    assign out[132:126] = v69[6:0];
    assign out[139:133] = v71[6:0];
    assign out[146:140] = v73[6:0];
    assign out[153:147] = v75[6:0];
    assign out[160:154] = v77[6:0];
    assign out[167:161] = v79[6:0];
    assign out[174:168] = v81[6:0];
    assign out[181:175] = v83[6:0];
    assign out[188:182] = v85[6:0];
    assign out[195:189] = v87[6:0];
    assign out[202:196] = v89[6:0];
    assign out[209:203] = v91[6:0];
    assign out[216:210] = v93[6:0];
    assign out[223:217] = v95[6:0];

    endmodule
