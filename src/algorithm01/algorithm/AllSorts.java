package algorithm01.algorithm;

import java.util.Arrays;

/**
 * all sorting method
 * <p>
 * Include bubble sort, select sort, insert sort, shell sort, quick sort, merge sort, heap sort
 * <p>
 * The input parameters of all methods are of type int, if you want to support all
 * types of input parameters, you can consider using generics.such as:
 * <pre>
 * @code
 * public static <T extends Comparable<\\? super T>> quickSort(T[] arr) {
 *   // ...
 * }
 * </pre>
 *
 * @author boomzy
 * @date 2021/1/30 16:25
 */
public class AllSorts {
    /**
     * bubble sort
     * <p>
     * The bubble sort algorithm is pairwise comparison. When the current element
     * is greater than the next element, the two elements are exchanged and the next
     * group is compared.
     * <p>
     * Each round of comparison will place the largest element in the array at the
     * end of the round, so the compared elements will be reduced after each cycle,
     * and finally the reduction is completed and the array is sorted successfully
     *
     * @param arr the array of before sort
     */
    public void bubbleSort(int[] arr) {
        if (arr == null || arr.length < 2) {
            return;
        }
        for (int end = arr.length - 1; end > 0; end--) {
            for (int i = 0; i < end; i++) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr, i, i + 1);
                }
            }
        }
    }

    /**
     * swap the two elements of array
     *
     * @param arr target array
     * @param i   the smaller index of array
     * @param j   the bigger index of array
     */
    private void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    /**
     * select sort
     * <p>
     * Selection sorting is a bit similar to bubble sorting, except that selection sorting
     * is performed every time after determining the minimum number of subscripts, which
     * greatly reduces the number of exchanges.
     *
     * @param arr the array of before sort
     */
    public void selectSort(int[] arr) {
        if (arr == null || arr.length < 2) {
            return;
        }
        for (int i = 0; i < arr.length - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] < arr[minIndex]) {
                    minIndex = j;
                }
            }
            swap(arr, i, minIndex);
        }
    }

    /**
     * insert sort
     * <p>
     * Insert sort algorithm is Each step inserts a piece of data to be sorted into the
     * previously sorted sequence until all elements are inserted.
     *
     * @param arr the array of before sort
     */
    public void insertSort(int[] arr) {
        if (arr == null || arr.length < 2) {
            return;
        }
        for (int i = 1; i < arr.length; i++) {
            for (int j = i - 1; j >= 0 && arr[j] > arr[j + 1]; j--) {
                swap(arr, j, j + 1);
            }
        }
    }

    /**
     * shell sort
     * <p>
     * Hill sort is an efficient implementation of insertion sort, also called reduced
     * incremental sort. In simple insertion sort, if the sequence to be sorted is in
     * positive order, the time complexity is O(n), if the sequence is basically ordered,
     * the efficiency of using direct insertion sort is very high. Hill sort takes advantage
     * of this feature.
     * <p>
     * The basic idea is: first divide the entire sequence of records to be sorted into several
     * sub-sequences for direct insertion sorting respectively, and then perform a direct
     * insertion sorting of all records when the records in the whole sequence are basically in order.
     *
     * @param arr the array of before sort
     */
    public void shellSort(int[] arr) {
        if (arr == null || arr.length < 2) {
            return;
        }
        int j;
        for (int gap = arr.length / 2; gap > 0; gap /= 2) {
            for (int i = gap; i < arr.length; i++) {
                int temp = arr[i];
                for (j = i; j >= gap && temp < arr[j - gap]; j -= gap) {
                    arr[j] = arr[j - gap];
                }
                arr[j] = temp;
            }
        }
    }

    /**
     * quick sort
     * <p>
     * Divide the sequence into left and right parts by sorting, where the value of the left
     * half is smaller than the value of the right half,
     * Then sort the left and right records separately until the entire sequence is in order.
     *
     * @param arr   the array of before sort
     * @param start the start index of array
     * @param end   the end index of array
     */
    public void quickSort(int[] arr, int start, int end) {
        if (start >= end) {
            return;
        }
        int low = start;
        int high = end;
        int stard = arr[start];
        while (low < high) {
            while (low < high && stard <= arr[high]) {
                high--;
            }
            arr[low] = arr[high];
            while (low < high && stard >= arr[low]) {
                low++;
            }
            arr[high] = arr[low];
        }
        arr[low] = stard;
        quickSort(arr, start, low);
        quickSort(arr, low + 1, end);
    }

    /**
     * merge sort
     * <p>
     * Divide the ordered list into two halves with the same number of elements as possible.
     * and Sort the two halves separately. finally Combine two ordered lists into one
     *
     * @param arr  the array of before sort
     * @param low  the start index of array
     * @param high the end index of array
     */
    public void mergeSort(int[] arr, int low, int high) {
        int middle = (low + high) / 2;
        if (low >= high) {
            return;
        }
        mergeSort(arr, low, middle);
        mergeSort(arr, middle + 1, high);
        merge(arr, low, middle, high);
    }

    /**
     * merge for merge sort
     *
     * @param arr the array of before sort
     * @param low the current start index of array
     * @param middle the current middle index of array
     * @param high the current end index of array
     */
    private void merge(int[] arr, int low, int middle, int high) {
        int[] temp = new int[high - low + 1];
        int i = low;
        int j = middle + 1;
        int index = 0;
        while (i <= middle && j <= high) {
            if (arr[i] < arr[j]) {
                temp[index++] = arr[i++];
            } else {
                temp[index++] = arr[j++];
            }

            while (i <= middle) {
                temp[index++] = arr[i++];
            }
            while (j <= high) {
                temp[index++] = arr[j++];
            }

            for (int k = 0; k < temp.length; k++) {
                arr[k + low] = temp[k];
            }
        }
    }
}
