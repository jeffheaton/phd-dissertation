package com.jeffheaton.dissertation.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by jeffh on 7/10/2016.
 */
public class ArrayUtils {
    public static List<String> string2list(String str) {
        List<String> result = new ArrayList<String>();
        String[] list = str.split(",");
        for(String s:list) {
            result.add(str);
        }
        return result;
    }

    public static String list2string(List<String> list) {
        StringBuilder result = new StringBuilder();
        for(String str:list) {
            if( result.length()!=0 ) {
                result.append(',');
            }
            result.append(str);
        }
        return result.toString();
    }
}

