package com.example.version1;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import static com.example.version1.R.id.button5;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    Button button;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 255.0f;
    private  static final int PERMISSION_REQUEST=0;
    private  static final int RESULT_LOAD = 1;
    private static final int RESULTS_TO_SHOW = 1;

    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    private Interpreter tflite;
    private List<String> labelList;

    private ByteBuffer imgData = null;
    //private float[][] labelProbArray = null;
    //private byte[][] labelProbArrayB = null;
    private float[][] labelProbArray = null;
//    s
    private int DIM_IMG_SIZE_X = 128;
    private int DIM_IMG_SIZE_Y = 128;
    private int DIM_PIXEL_SIZE = 1;

    private int[] intValues;

    private Button classify_button;
    private TextView label1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    PERMISSION_REQUEST);
        }

        imageView = (ImageView) findViewById(R.id.imageView);
        button = (Button) findViewById(R.id.button4);

        button.setOnClickListener(new View.OnClickListener(){

            @Override
            public  void onClick(View v){
                Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i,RESULT_LOAD);
            }
        });

        intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

//        super.onCreate(savedInstanceState);

        //initilize graph and labels
        try{
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
            labelList = loadLabelList();
        } catch (Exception ex){
            ex.printStackTrace();
        }

//        imgData = ByteBuffer.allocateDirect(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData = ByteBuffer.allocateDirect(4*DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());

        labelProbArray = new float[1][labelList.size()];
//        labelProbArrayB= new byte[1][labelList.size()];
//        setContentView(R.layout.activity_classify);

        classify_button = (Button)findViewById(button5);
        classify_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // get current bitmap from imageView
                Bitmap bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
                // resize the bitmap to the required input size to the CNN
//                Bitmap bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
                // convert bitmap to byte array
                convertBitmapToByteBuffer(bitmap);
                // pass byte data to the graph
                tflite.run(imgData, labelProbArray);
                for(int i=0;i<3;i++) {
                    System.out.println("label" + labelProbArray[0][i]);
                }
                // display the results
                printTopKLabels();
            }
        });


    }


    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("converted_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        System.out.println("ImageData"+imgData);


        imgData.rewind();
        //imgData
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                //System.out.println("okokokokok--------" + val);


                // get rgb values from intValues where each int holds the rgb values for a pixel.
//                // if quantized, convert each rgb value to a byte, otherwise to a float
//                imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//                imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//                imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//                imgData.putFloat((((val >> 16) & 0xFF))/255f);
//                imgData.putFloat((((val >> 8) & 0xFF))/255f);
//                imgData.putFloat((val)/255f);
                float rChannel = (val >> 16) & 0xFF;
                float gChannel = (val >> 8) & 0xFF;
                float bChannel = (val) & 0xFF;
                float pixelValue = (rChannel + gChannel + bChannel) / 3 / 255.f;
                imgData.putFloat(pixelValue);
                }

            }
        }

    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(this.getAssets().open("label.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }




    @Override
    public void onRequestPermissionsResult(final int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST) {
            if (!(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                Toast.makeText(getApplicationContext(),"This application needs read, write, and camera permissions to run. Application now closing.",Toast.LENGTH_LONG).show();
                System.exit(0);
            }
        }
    }


    private void printTopKLabels() {
        // add all results to priority queue
        int mx=0;
        for (int i = 0; i < labelList.size(); ++i) {

            if (labelProbArray[0][i] > labelProbArray[0][mx]) {
                mx = i;
            }
        }
        String x=labelList.get(mx);
        System.out.println(x);

//            sortedLabels.add(
//                    new AbstractMap.SimpleEntry<>(labelList.get(i), (labelProbArrayB[0][i] & 0xff) / 255.0f));
//            sortedLabels.add(
//                    new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
//            if (sortedLabels.size() > RESULTS_TO_SHOW) {
//                sortedLabels.poll();
//            }
//        }

        // get top results from priority queue
//        final int size = sortedLabels.size();
//        for (int i = 0; i < size; ++i) {
//            Map.Entry<String, Float> label = sortedLabels.poll();
//            String str = label.getKey();
//            System.out.println("str"+str);
//            System.out.println(str.getClass().getName());
//            topLables.add(str);
//            topConfidence[i] = String.format("%.0f%%",label.getValue()*100);
//        }

        // set the corresponding textviews with the results
        label1=(TextView)findViewById(R.id.textView);
        label1.setText("1. "+(String)(x));

    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode){
            case RESULT_LOAD:
                if(resultCode == RESULT_OK){
                    Uri selectedImage = data.getData();
                    String[] filePathColumn = {MediaStore.Images.Media.DATA};
                    Cursor cursor = getContentResolver().query(selectedImage,filePathColumn,null,null,null);
                    cursor.moveToFirst();
                    int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                    String picturePath = cursor.getString(columnIndex);
                    cursor.close();
                    imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));
                }

        }
    }
}
