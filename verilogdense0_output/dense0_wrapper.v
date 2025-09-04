`timescale 1 ns / 1 ps

module dense0_wrapper (
   input clk,
    // verilator lint_off UNUSEDSIGNAL
    input [511:0] inp,
    // verilator lint_on UNUSEDSIGNAL
    output [1503:0] out
);
    wire [511:0] packed_inp;
    wire [1394:0] packed_out;

    assign packed_inp[511:0] = inp[511:0];

    dense0 op (
        .clk(clk),
        .inp(packed_inp),
        .out(packed_out)
    );

    assign out[4:0] = 5'b0;
    assign out[46:5] = packed_out[41:0];
    assign out[50:47] = 4'b0;
    assign out[93:51] = packed_out[84:42];
    assign out[94:94] = 1'b0;
    assign out[140:95] = packed_out[130:85];
    assign out[143:141] = 3'b0;
    assign out[187:144] = packed_out[174:131];
    assign out[191:188] = 4'b0;
    assign out[281:192] = packed_out[264:175];
    assign out[284:282] = 3'b0;
    assign out[328:285] = packed_out[308:265];
    assign out[332:329] = 4'b0;
    assign out[374:333] = packed_out[350:309];
    assign out[375:375] = {1{packed_out[350]}};
    assign out[379:376] = 4'b0;
    assign out[422:380] = packed_out[393:351];
    assign out[424:423] = 2'b0;
    assign out[469:425] = packed_out[438:394];
    assign out[473:470] = 4'b0;
    assign out[516:474] = packed_out[481:439];
    assign out[520:517] = 4'b0;
    assign out[563:521] = packed_out[524:482];
    assign out[610:568] = packed_out[567:525];
    assign out[567:564] = 4'b0;
    assign out[611:611] = 1'b0;
    assign out[657:612] = packed_out[613:568];
    assign out[704:660] = packed_out[658:614];
    assign out[659:658] = 2'b0;
    assign out[751:708] = packed_out[702:659];
    assign out[707:705] = 3'b0;
    assign out[798:756] = packed_out[745:703];
    assign out[755:752] = 4'b0;
    assign out[845:802] = packed_out[789:746];
    assign out[801:799] = 3'b0;
    assign out[892:852] = packed_out[830:790];
    assign out[851:846] = 6'b0;
    assign out[939:897] = packed_out[873:831];
    assign out[896:893] = 4'b0;
    assign out[986:944] = packed_out[916:874];
    assign out[943:940] = 4'b0;
    assign out[1033:991] = packed_out[959:917];
    assign out[990:987] = 4'b0;
    assign out[1080:1039] = packed_out[1001:960];
    assign out[1038:1034] = 5'b0;
    assign out[1127:1084] = packed_out[1045:1002];
    assign out[1083:1081] = 3'b0;
    assign out[1174:1132] = packed_out[1088:1046];
    assign out[1131:1128] = 4'b0;
    assign out[1221:1178] = packed_out[1132:1089];
    assign out[1268:1225] = packed_out[1176:1133];
    assign out[1177:1175] = 3'b0;
    assign out[1315:1274] = packed_out[1218:1177];
    assign out[1224:1222] = 3'b0;
    assign out[1362:1318] = packed_out[1263:1219];
    assign out[1273:1269] = 5'b0;
    assign out[1409:1367] = packed_out[1306:1264];
    assign out[1317:1316] = 2'b0;
    assign out[1456:1413] = packed_out[1350:1307];
    assign out[1366:1363] = 4'b0;
    assign out[1503:1460] = packed_out[1394:1351];
    assign out[1412:1410] = 3'b0;
    assign out[1459:1457] = 3'b0;

endmodule
