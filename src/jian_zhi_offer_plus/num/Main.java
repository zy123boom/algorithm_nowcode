package jian_zhi_offer_plus.num;

import java.util.Objects;

/**
 * 第1章 整数
 * <p>
 * 来源：力扣（LeetCode）
 * 链接：https://leetcode-cn.com/problem-list/e8X3pBZi/
 *
 * @author boomzy
 * @date 2021/12/18 18:54
 */
public class Main {

    /**
     * 1.整数相除
     * <p>
     * 给定两个整数 a 和 b ，求它们的除法的商 a/b ，要求不得使用乘号 '*'、除号 '/' 以及求余符号 '%' 。
     * <p>
     * 注意：
     * 整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2
     * 假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231, 231−1]。本题中，如果除法结果溢出，则返回 231 − 1
     * <p>
     * 示例 1：
     * 输入：a = 15, b = 2
     * 输出：7
     * 解释：15/2 = truncate(7.5) = 7
     * <p>
     * 示例 2：
     * 输入：a = 7, b = -3
     * 输出：-2
     * 解释：7/-3 = truncate(-2.33333..) = -2
     * <p>
     * 示例 3：
     * 输入：a = 0, b = 1
     * 输出：0
     * <p>
     * 示例 4：
     * 输入：a = 1, b = 1
     * 输出：1
     *
     * @param a
     * @param b
     * @return
     */
    public int divide(int a, int b) {
        /*
            基于减法实现除法的思路
            不断用被除数减去除数，复杂度O(n)
            优化：
            当被除数大于除数时，继续比较判断被除数书否大于除数的2倍，如果是，则继续判断被除数
            是否大于除数的4倍、8倍等...如果被除数最多大于除数的2的k次方倍，则将被除数减去2的k次方倍。
            （减去的是除数的几倍最后的商就是几，循环步骤后累加的商就是最终结果）
            然后将剩余的被除数重复前面的步骤。
            由于每次将除数翻倍，时间复杂度O(logN)
         */
        // 边界溢出条件，超出了int最大范围
        if (a == Integer.MIN_VALUE && b == -1) {
            return Integer.MAX_VALUE;
        }

        // 正负数判断与参数取反
        // 正负数判断目的是如果参数有一个为负数，则需要取反计算结果
        // 参数取反目的是可以转化为正数方便后续运算
        int negative = 2;
        if (a > 0) {
            negative--;
            a = -a;
        }
        if (b > 0) {
            negative--;
            b = -b;
        }
        int result = divideCore(a, b);
        return negative == 1 ? -result : result;
    }

    private int divideCore(int dividend, int divisor) {
        int result = 0;
        while (dividend <= divisor) {
            // 不断乘2倍
            int value = divisor;
            // 当前商
            int quotient = 1;
            while (value >= Integer.MIN_VALUE / 2 && dividend <= value + value) {
                quotient += quotient;
                value += value;
            }
            // 当前商不断累加就是最终结果
            result += quotient;
            dividend -= value;
        }
        return result;
    }

    /**
     * 2.二进制加法
     * <p>
     * 给定两个 01 字符串 a 和 b ，请计算它们的和，并以二进制字符串的形式输出。
     * 输入为 非空 字符串且只包含数字 1 和 0。
     * <p>
     * 示例 1:
     * 输入: a = "11", b = "10"
     * 输出: "101"
     * <p>
     * 示例 2:
     * 输入: a = "1010", b = "1011"
     * 输出: "10101"
     *  
     * 提示：
     * 每个字符串仅由字符 '0' 或 '1' 组成。
     * 1 <= a.length, b.length <= 10^4
     * 字符串如果不是 "0" ，就都不含前导零。
     *
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        if (a == null || "".equals(a) || b == null || "".equals(b)) {
            return "";
        }

        if (a.length() > Math.pow(10, 4) || b.length() > Math.pow(10, 4)) {
            return "";
        }

        /*
            从字符串右端开始累加，逢2进1
         */
        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;
        // 进位
        int carry = 0;
        while (i >= 0 || j >= 0) {
            // a和b当前位置上的数
            int digitA = i >= 0 ? a.charAt(i--) - '0' : 0;
            int digitB = j >= 0 ? b.charAt(j--) - '0' : 0;
            int sum = digitA + digitB + carry;
            carry = sum >= 2 ? 1 : 0;
            sum = sum >= 2 ? sum - 2 : sum;
            sb.append(sum);
        }
        if (carry == 1) {
            sb.append(carry);
        }
        // 由于是从右向左，最低位在最左边，最高位在最右边。需要取反
        return sb.reverse().toString();
    }
}
