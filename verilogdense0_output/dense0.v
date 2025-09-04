`timescale 1 ns / 1 ps

module dense0 (
    input clk,
    input [511:0] inp,
    output reg [1394:0] out
);

    reg [512-1:0] stage0_inp;
    wire [1395-1:0] stage0_out;

    dense0_stage0 stage0 (.inp(stage0_inp), .out(stage0_out));

    always @(posedge clk) begin
        stage0_inp <= inp;
        out <= stage0_out;
    end
endmodule
