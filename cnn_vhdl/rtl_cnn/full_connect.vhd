-- 全连接层
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_signed.all;

entity full_connect is
	port(
		clk:in std_logic;
		rst_n:in std_logic;
		data_en:in std_logic;  
		data1,data2,data3,data4:in std_logic_vector(7 downto 0); 

		cnn_finish:out std_logic;
		result:out std_logic_vector(3 downto 0)
	);
end entity;

architecture behav of full_connect is

constant w0_1:std_logic_vector(1 downto 0):="11";constant w0_9:std_logic_vector(1 downto 0):="01";
constant w0_2:std_logic_vector(1 downto 0):="01";constant w0_10:std_logic_vector(1 downto 0):="01";
constant w0_3:std_logic_vector(1 downto 0):="00";constant w0_11:std_logic_vector(1 downto 0):="01";
constant w0_4:std_logic_vector(1 downto 0):="01";constant w0_12:std_logic_vector(1 downto 0):="11";
constant w0_5:std_logic_vector(1 downto 0):="00";constant w0_13:std_logic_vector(1 downto 0):="01";
constant w0_6:std_logic_vector(1 downto 0):="01";constant w0_14:std_logic_vector(1 downto 0):="01";
constant w0_7:std_logic_vector(1 downto 0):="01";constant w0_15:std_logic_vector(1 downto 0):="11";
constant w0_8:std_logic_vector(1 downto 0):="01";constant w0_16:std_logic_vector(1 downto 0):="01";

constant w1_1:std_logic_vector(1 downto 0):="01";constant w1_9:std_logic_vector(1 downto 0):="01";
constant w1_2:std_logic_vector(1 downto 0):="01";constant w1_10:std_logic_vector(1 downto 0):="01";
constant w1_3:std_logic_vector(1 downto 0):="00";constant w1_11:std_logic_vector(1 downto 0):="01";
constant w1_4:std_logic_vector(1 downto 0):="01";constant w1_12:std_logic_vector(1 downto 0):="11";
constant w1_5:std_logic_vector(1 downto 0):="01";constant w1_13:std_logic_vector(1 downto 0):="01";
constant w1_6:std_logic_vector(1 downto 0):="00";constant w1_14:std_logic_vector(1 downto 0):="01";
constant w1_7:std_logic_vector(1 downto 0):="01";constant w1_15:std_logic_vector(1 downto 0):="00";
constant w1_8:std_logic_vector(1 downto 0):="01";constant w1_16:std_logic_vector(1 downto 0):="01";

constant w2_1:std_logic_vector(1 downto 0):="01";constant w2_9:std_logic_vector(1 downto 0):="01";
constant w2_2:std_logic_vector(1 downto 0):="01";constant w2_10:std_logic_vector(1 downto 0):="01";
constant w2_3:std_logic_vector(1 downto 0):="11";constant w2_11:std_logic_vector(1 downto 0):="11";
constant w2_4:std_logic_vector(1 downto 0):="01";constant w2_12:std_logic_vector(1 downto 0):="01";
constant w2_5:std_logic_vector(1 downto 0):="01";constant w2_13:std_logic_vector(1 downto 0):="01";
constant w2_6:std_logic_vector(1 downto 0):="01";constant w2_14:std_logic_vector(1 downto 0):="11";
constant w2_7:std_logic_vector(1 downto 0):="11";constant w2_15:std_logic_vector(1 downto 0):="01";
constant w2_8:std_logic_vector(1 downto 0):="01";constant w2_16:std_logic_vector(1 downto 0):="01";

constant w3_1:std_logic_vector(1 downto 0):="01";constant w3_9:std_logic_vector(1 downto 0):="01";
constant w3_2:std_logic_vector(1 downto 0):="00";constant w3_10:std_logic_vector(1 downto 0):="01";
constant w3_3:std_logic_vector(1 downto 0):="01";constant w3_11:std_logic_vector(1 downto 0):="01";
constant w3_4:std_logic_vector(1 downto 0):="01";constant w3_12:std_logic_vector(1 downto 0):="11";
constant w3_5:std_logic_vector(1 downto 0):="11";constant w3_13:std_logic_vector(1 downto 0):="01";
constant w3_6:std_logic_vector(1 downto 0):="01";constant w3_14:std_logic_vector(1 downto 0):="00";
constant w3_7:std_logic_vector(1 downto 0):="01";constant w3_15:std_logic_vector(1 downto 0):="11";
constant w3_8:std_logic_vector(1 downto 0):="01";constant w3_16:std_logic_vector(1 downto 0):="01";

constant w4_1:std_logic_vector(1 downto 0):="01";constant w4_9:std_logic_vector(1 downto 0):="01";
constant w4_2:std_logic_vector(1 downto 0):="11";constant w4_10:std_logic_vector(1 downto 0):="01";
constant w4_3:std_logic_vector(1 downto 0):="01";constant w4_11:std_logic_vector(1 downto 0):="11";
constant w4_4:std_logic_vector(1 downto 0):="01";constant w4_12:std_logic_vector(1 downto 0):="01";
constant w4_5:std_logic_vector(1 downto 0):="01";constant w4_13:std_logic_vector(1 downto 0):="01";
constant w4_6:std_logic_vector(1 downto 0):="01";constant w4_14:std_logic_vector(1 downto 0):="01";
constant w4_7:std_logic_vector(1 downto 0):="01";constant w4_15:std_logic_vector(1 downto 0):="01";
constant w4_8:std_logic_vector(1 downto 0):="01";constant w4_16:std_logic_vector(1 downto 0):="01";

constant w5_1:std_logic_vector(1 downto 0):="01";constant w5_9:std_logic_vector(1 downto 0):="01";
constant w5_2:std_logic_vector(1 downto 0):="11";constant w5_10:std_logic_vector(1 downto 0):="01";
constant w5_3:std_logic_vector(1 downto 0):="01";constant w5_11:std_logic_vector(1 downto 0):="01";
constant w5_4:std_logic_vector(1 downto 0):="01";constant w5_12:std_logic_vector(1 downto 0):="00";
constant w5_5:std_logic_vector(1 downto 0):="01";constant w5_13:std_logic_vector(1 downto 0):="01";
constant w5_6:std_logic_vector(1 downto 0):="11";constant w5_14:std_logic_vector(1 downto 0):="01";
constant w5_7:std_logic_vector(1 downto 0):="01";constant w5_15:std_logic_vector(1 downto 0):="01";
constant w5_8:std_logic_vector(1 downto 0):="01";constant w5_16:std_logic_vector(1 downto 0):="01";

constant w6_1:std_logic_vector(1 downto 0):="01";constant w6_9:std_logic_vector(1 downto 0):="01";
constant w6_2:std_logic_vector(1 downto 0):="01";constant w6_10:std_logic_vector(1 downto 0):="01";
constant w6_3:std_logic_vector(1 downto 0):="11";constant w6_11:std_logic_vector(1 downto 0):="01";
constant w6_4:std_logic_vector(1 downto 0):="01";constant w6_12:std_logic_vector(1 downto 0):="01";
constant w6_5:std_logic_vector(1 downto 0):="01";constant w6_13:std_logic_vector(1 downto 0):="01";
constant w6_6:std_logic_vector(1 downto 0):="01";constant w6_14:std_logic_vector(1 downto 0):="01";
constant w6_7:std_logic_vector(1 downto 0):="11";constant w6_15:std_logic_vector(1 downto 0):="01";
constant w6_8:std_logic_vector(1 downto 0):="01";constant w6_16:std_logic_vector(1 downto 0):="11";

constant w7_1:std_logic_vector(1 downto 0):="01";constant w7_9:std_logic_vector(1 downto 0):="01";
constant w7_2:std_logic_vector(1 downto 0):="01";constant w7_10:std_logic_vector(1 downto 0):="01";
constant w7_3:std_logic_vector(1 downto 0):="11";constant w7_11:std_logic_vector(1 downto 0):="11";
constant w7_4:std_logic_vector(1 downto 0):="01";constant w7_12:std_logic_vector(1 downto 0):="01";
constant w7_5:std_logic_vector(1 downto 0):="01";constant w7_13:std_logic_vector(1 downto 0):="01";
constant w7_6:std_logic_vector(1 downto 0):="01";constant w7_14:std_logic_vector(1 downto 0):="01";
constant w7_7:std_logic_vector(1 downto 0):="00";constant w7_15:std_logic_vector(1 downto 0):="00";
constant w7_8:std_logic_vector(1 downto 0):="01";constant w7_16:std_logic_vector(1 downto 0):="01";

constant w8_1:std_logic_vector(1 downto 0):="01";constant w8_9:std_logic_vector(1 downto 0):="01";
constant w8_2:std_logic_vector(1 downto 0):="01";constant w8_10:std_logic_vector(1 downto 0):="01";
constant w8_3:std_logic_vector(1 downto 0):="11";constant w8_11:std_logic_vector(1 downto 0):="00";
constant w8_4:std_logic_vector(1 downto 0):="01";constant w8_12:std_logic_vector(1 downto 0):="01";
constant w8_5:std_logic_vector(1 downto 0):="01";constant w8_13:std_logic_vector(1 downto 0):="11";
constant w8_6:std_logic_vector(1 downto 0):="11";constant w8_14:std_logic_vector(1 downto 0):="01";
constant w8_7:std_logic_vector(1 downto 0):="01";constant w8_15:std_logic_vector(1 downto 0):="01";
constant w8_8:std_logic_vector(1 downto 0):="01";constant w8_16:std_logic_vector(1 downto 0):="01";

constant w9_1:std_logic_vector(1 downto 0):="11";constant w9_9:std_logic_vector(1 downto 0):="01";
constant w9_2:std_logic_vector(1 downto 0):="11";constant w9_10:std_logic_vector(1 downto 0):="11";
constant w9_3:std_logic_vector(1 downto 0):="00";constant w9_11:std_logic_vector(1 downto 0):="00";
constant w9_4:std_logic_vector(1 downto 0):="11";constant w9_12:std_logic_vector(1 downto 0):="01";
constant w9_5:std_logic_vector(1 downto 0):="11";constant w9_13:std_logic_vector(1 downto 0):="11";
constant w9_6:std_logic_vector(1 downto 0):="11";constant w9_14:std_logic_vector(1 downto 0):="11";
constant w9_7:std_logic_vector(1 downto 0):="11";constant w9_15:std_logic_vector(1 downto 0):="11";
constant w9_8:std_logic_vector(1 downto 0):="11";constant w9_16:std_logic_vector(1 downto 0):="01";

component multi is
	port(
		ready:in std_logic;
		d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16:in std_logic_vector(9 downto 0);
		w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16:in std_logic_vector(1 downto 0);
		data_out:out std_logic_vector(11 downto 0)
	);
end component;

component full_sipo is
	port(
		clk:in std_logic;
		rst_n:in std_logic;
		ready:in std_logic;
		data1,data2,data3,data4:in std_logic_vector(7 downto 0);
		d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16:out std_logic_vector(7 downto 0);
		finish:out std_logic
	);
	
end component;


signal o0,o1,o2,o3,o4,o5,o6,o7,o8,o9:std_logic_vector(11 downto 0);
signal a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16: std_logic_vector(9 downto 0);
signal d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16: std_logic_vector(7 downto 0);
signal cnt:std_logic_vector(3 downto 0);
signal oo1,oo2,oo3,oo4,oo5:std_logic_vector(11 downto 0);
signal cnt6,cnt7:std_logic;
signal ooo1,ooo2:std_logic_vector(11 downto 0);
signal fini1,fini2,fini3:std_logic;
signal ready:std_logic; -- SIPO完成的结束信号，用于启动后续乘法器


begin

a1<="00"& d1 when d1(7)='0' else "11"&d1;
a2<="00"& d2 when d2(7)='0' else "11"&d2;
a3<="00"& d3 when d3(7)='0' else "11"&d3;
a4<="00"& d4 when d4(7)='0' else "11"&d4;
a5<="00"& d5 when d5(7)='0' else "11"&d5;
a6<="00"& d6 when d6(7)='0' else "11"&d6;
a7<="00"& d7 when d7(7)='0' else "11"&d7;
a8<="00"& d8 when d8(7)='0' else "11"&d8;
a9<="00"& d9 when d9(7)='0' else "11"&d9;
a10<="00"& d10 when d10(7)='0' else "11"&d10;
a11<="00"& d11 when d11(7)='0' else "11"&d11;
a12<="00"& d12 when d12(7)='0' else "11"&d12;
a13<="00"& d13 when d13(7)='0' else "11"&d13;
a14<="00"& d14 when d14(7)='0' else "11"&d14;
a15<="00"& d15 when d15(7)='0' else "11"&d15;
a16<="00"& d16 when d16(7)='0' else "11"&d16;


c1:full_sipo
	port map(clk,rst_n,data_en,data1,data2,data3,data4,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,
	d11,d12,d13,d14,d15,d16,ready);


u0:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w0_1,w0_2,
	w0_3,w0_4,w0_5,w0_6,w0_7,w0_8,w0_9,w0_10,w0_11,w0_12,w0_13,w0_14,w0_15,w0_16,o0);


u1:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w1_1,w1_2,
	w1_3,w1_4,w1_5,w1_6,w1_7,w1_8,w1_9,w1_10,w1_11,w1_12,w1_13,w1_14,w1_15,w1_16,o1);

u2:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w2_1,w2_2,
	w2_3,w2_4,w2_5,w2_6,w2_7,w2_8,w2_9,w2_10,w2_11,w2_12,w2_13,w2_14,w2_15,w2_16,o2);
	
u3:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w3_1,w3_2,
	w3_3,w3_4,w3_5,w3_6,w3_7,w3_8,w3_9,w3_10,w3_11,w3_12,w3_13,w3_14,w3_15,w3_16,o3);
	
u4:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w4_1,w4_2,
	w4_3,w4_4,w4_5,w4_6,w4_7,w4_8,w4_9,w4_10,w4_11,w4_12,w4_13,w4_14,w4_15,w4_16,o4);
	
u5:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w5_1,w5_2,
	w5_3,w5_4,w5_5,w5_6,w5_7,w5_8,w5_9,w5_10,w5_11,w5_12,w5_13,w5_14,w5_15,w5_16,o5);
	
u6:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w6_1,w6_2,
	w6_3,w6_4,w6_5,w6_6,w6_7,w6_8,w6_9,w6_10,w6_11,w6_12,w6_13,w6_14,w6_15,w6_16,o6);
	
u7:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w7_1,w7_2,
	w7_3,w7_4,w7_5,w7_6,w7_7,w7_8,w7_9,w7_10,w7_11,w7_12,w7_13,w7_14,w7_15,w7_16,o7);
	
u8:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w8_1,w8_2,
	w8_3,w8_4,w8_5,w8_6,w8_7,w8_8,w8_9,w8_10,w8_11,w8_12,w8_13,w8_14,w8_15,w8_16,o8);
	
u9:multi
	port map(ready,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,w9_1,w9_2,
	w9_3,w9_4,w9_5,w9_6,w9_7,w9_8,w9_9,w9_10,w9_11,w9_12,w9_13,w9_14,w9_15,w9_16,o9);
	
process(o0,o1,clk)
begin
	if(o0>o1)then
	
		oo1<=o0;
	else
	
		oo1<=o1;
	end if;
end process;

process(o2,o3,clk)
begin
	if(o2>o3)then
	
		oo2<=o2;
	else
	
		oo2<=o3;
	end if;
end process;

process(o4,o5,clk)
begin
	if(o4>o5)then
	
		oo3<=o4;
	else
	
		oo3<=o5;
	end if;
end process;

process(o6,o7,clk)
begin
	if(o6>o7)then

		oo4<=o6;
	else

		oo4<=o7;
	end if;
end process;

process(o8,o9,clk)
begin
	if(o8>o9)then

		oo5<=o8;
	else

		oo5<=o9;
	end if;
end process;
		
process(oo1,oo2)
begin
	if(oo1>oo2)then
	
		ooo1<=oo1;
	else
	
		ooo1<=oo2;
	end if;
end process;

process(oo3,oo4)
begin
	if(oo3>oo4)then
	
		ooo2<=oo3;
	else
		
		ooo2<=oo4;
	end if;
end process;

process(ooo1,ooo2,oo5)
begin
	if(ooo1>=oo5 and ooo1>=ooo2)then
		if(oo1>oo2)then
			if(o0>o1)then
				cnt<="0000";
			else
				cnt<="0001";
			end if;
		else
			if(o2>o3)then
				cnt<="0010";
			else
				cnt<="0011";
			end if;
		end if;
	elsif(ooo2>=oo5 and ooo2>=ooo1)then
		if(oo3>oo4)then
			if(o4>o5)then
				cnt<="0100";
			else
				cnt<="0101";
			end if;
		else
			if(o6>o7)then
				cnt<="0110";
			else
				cnt<="0111";
			end if;
		end if;
	elsif(oo5>=ooo1 and oo5>=ooo2)then
		if(o8>o9)then
			cnt<="1000";
		else
			cnt<="1001";
		end if;
	end if;
end process;

process(clk,rst_n)
begin
	if(rst_n='0')then
		fini1<='0';
	elsif(rising_edge(clk))then
		fini1<=ready;
		fini2<=fini1;
		fini3<=fini2;
		cnn_finish<=fini3;
	end if;
end process;

result<=cnt;

end behav;




----------------------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_signed.all;

entity multi is
	port(
		ready:in std_logic;
		d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16:in std_logic_vector(9 downto 0);
		w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16:in std_logic_vector(1 downto 0);
		data_out:out std_logic_vector(11 downto 0)
	);
end entity;


architecture behav of multi is
begin

process(ready)
begin
	if(rising_edge(ready))then
		data_out<=d1*w1+d2*w2+d3*w3+d4*w4+d5*w5+d6*w6+d7*w7+d8*w8+d9*w9+d10*w10+d11*w11+d12*w12+d13*w13+d14*w14+d15*w15+d16*w16;
	end if;
end process;

end behav;	


--------------------------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_signed.all;


entity full_sipo is
	port(
		clk:in std_logic;
		rst_n:in std_logic;
		ready:in std_logic;
		data1,data2,data3,data4:in std_logic_vector(7 downto 0);
		d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16:out std_logic_vector(7 downto 0);
		finish:out std_logic
	);
	
end entity;

architecture behav of full_sipo is

signal a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16:std_logic_vector(7 downto 0);
signal cnt:std_logic_vector(1 downto 0);
signal co:std_logic;
signal co1:std_logic;

begin

process(rst_n,ready) --计数器4次
begin
	if(rst_n='0')then
		cnt<="00";
	elsif(rising_edge(ready))then
		if(cnt="11")then
			cnt<="00";
			co<='1';
		else
			co<='0';
		end if;
		cnt<=cnt+'1';
	end if;
end process;

process(cnt,clk) --暂存
begin
	case cnt is
	when "00" =>a1<=data1;a2<=data2;a3<=data3;a4<=data4;
	when "01" =>a5<=data1;a6<=data2;a7<=data3;a8<=data4;
	when "10" =>a9<=data1;a10<=data2;a11<=data3;a12<=data4;
	when "11" =>a13<=data1;a14<=data2;a15<=data3;a16<=data4;
	when others=>null;
	end case;
end process;

process(co)
begin
	if(rising_edge(co))then
		d1<=a1;d2<=a2;d3<=a3;d4<=a4;
		d5<=a5;d6<=a6;d7<=a7;d8<=a8;
		d9<=a9;d10<=a10;d11<=a11;d12<=a12;
		d13<=a13;d14<=a14;d15<=a15;d16<=a16;
	end if;
end process;

process(clk,rst_n)
begin
	if(rst_n='0')then
		co1<='0';
	elsif(rising_edge(clk))then
		co1<=co;
		finish<=co1;
	end if;
end process;


end behav;
