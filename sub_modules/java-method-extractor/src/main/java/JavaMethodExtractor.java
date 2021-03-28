import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.kohsuke.args4j.CmdLineException;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import static java.lang.Math.ceil;

public class JavaMethodExtractor {
    /**
     * Adapted from code2seq.
     * Extracts Java Methods from the raw .java files datasets that they collected (java-small, java-medium, java-large)
     * Extracted methods are stored into a .json file at a specified location.
     * The reason we have this is to align the preprocessing of code2seq with ours, i.e., that we can use our whole
     * pipeline on their raw data.
     *
     * Build file with
     * mvn clean compile assembly:single
     *
     * Usage:
     * java -jar JavaMethodExtractor-1.0.0-SNAPSHOT.jar --dir {RAW_CODE2SEQ_JAVA_FILES} --output_dir {OUTPUT_DIR}
     */

    private static CommandLineValues s_CommandLineValues;

    public static void waitForHealthState(ZKFailoverController zkfc,
                                          HealthMonitor.State state,
                                          MultithreadedTestUtil.TestContext ctx)
            throws Exception {
        while (zkfc.getLastHealthState() != state) {
            if (ctx != null) {
                ctx.checkException();
            }
            Thread.sleep(50);
        }
    }

    public static void main(String[] args) {
        try {
            s_CommandLineValues = new CommandLineValues(args);
        } catch (CmdLineException e) {
            e.printStackTrace();
            return;
        }

        if (s_CommandLineValues.File != null) {
            ExtractMethodsTask extractMethodsTask = new ExtractMethodsTask(s_CommandLineValues,
                    s_CommandLineValues.File.toPath());
            List<MethodContent> methodContents = extractMethodsTask.processFile();
            outputMethodContentsAsJson(methodContents);
        } else if (s_CommandLineValues.Dir != null) {
            extractDir();
        }
    }

    private static void extractDir() {
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(s_CommandLineValues.NumThreads);
        LinkedList<ExtractMethodsTask> tasks = new LinkedList<>();
        try {
            Files.walk(Paths.get(s_CommandLineValues.Dir)).filter(Files::isRegularFile)
                    .filter(p -> p.toString().toLowerCase().endsWith(".java")).forEach(f -> {
                ExtractMethodsTask task = new ExtractMethodsTask(s_CommandLineValues, f);
                tasks.add(task);
            });
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        List<Future<List<MethodContent>>> tasksResults = null;
        try {
            tasksResults = executor.invokeAll(tasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
        List<MethodContent> allMethodContents = new ArrayList<>();
        tasksResults.forEach(f -> {
            try {
                allMethodContents.addAll(f.get());
            } catch (InterruptedException | ExecutionException e) {
                //e.printStackTrace();
                System.err.println(e.getMessage());
            }
        });

        outputMethodContentsAsJson(allMethodContents);
    }

    private static void outputMethodContentsAsJson(List<MethodContent> methodContents) {
        Gson gson = new GsonBuilder().disableHtmlEscaping().create();
        if (s_CommandLineValues.OutputDir != null) {
            int dataset_slice_size = 500000;
            int num_dataset_slices = (int)ceil(((double) methodContents.size()) / dataset_slice_size);
            for (int i = 0; i < num_dataset_slices; i++) {
                int toIndex = (i + 1) * dataset_slice_size > methodContents.size() ? methodContents.size() : (i + 1) * dataset_slice_size;
                List<MethodContent> dataset_slice = methodContents.subList(i * dataset_slice_size, toIndex);
                String jsonString = gson.toJson(dataset_slice);
                String fileName = String.format("%s/dataset-%d.json", s_CommandLineValues.OutputDir, i);
                File file = new File(fileName);
                file.getParentFile().mkdirs();
                try {
                    FileWriter writer = new FileWriter(file);
                    writer.write(jsonString);
                    writer.flush();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

        } else {
            String jsonString = gson.toJson(methodContents);
            System.out.println(jsonString);
        }
    }

}
