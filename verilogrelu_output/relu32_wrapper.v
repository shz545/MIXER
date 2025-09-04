`timescale 1 ns / 1 ps

module relu32_wrapper (
    // verilator lint_off UNUSEDSIGNAL
    input [255:0] inp,
    // verilator lint_on UNUSEDSIGNAL
    output [223:0] out
);
    wire [255:0] packed_inp;
    wire [223:0] packed_out;

    assign packed_inp[255:0] = inp[255:0];

    relu32 op (
        .inp(packed_inp),
        .out(packed_out)
    );

    assign out[223:0] = packed_out[223:0];

endmodule
