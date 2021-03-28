import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

public class ExtractMethodsTask implements Callable<List<MethodContent>> {

    private final CommandLineValues commandLineValues;
    private final Path filePath;

    public ExtractMethodsTask(CommandLineValues commandLineValues, Path path) {
        this.commandLineValues = commandLineValues;
        this.filePath = path;
    }

    @Override
    public List<MethodContent> call() throws Exception {
        return processFile();
    }

    public List<MethodContent> processFile() {
        String code;

        try {
            if (commandLineValues.MaxFileLength > 0 &&
                    Files.lines(filePath, Charset.defaultCharset()).count() > commandLineValues.MaxFileLength) {
                return new ArrayList<>();
            }
        } catch (IOException e) {
            return new ArrayList<>();
        }
        try {
            code = new String(Files.readAllBytes(filePath));
        } catch (IOException e) {
            e.printStackTrace();
            code = "";
        }

        CompilationUnit compilationUnit = parseFileWithRetries(code);
        MethodVisitor methodVisitor = new MethodVisitor(commandLineValues);

        methodVisitor.visit(compilationUnit, null);

        return methodVisitor.getMethods();
    }

    private CompilationUnit parseFileWithRetries(String code) {
        final String classPrefix = "public class Test {";
        final String classSuffix = "}";
        final String methodPrefix = "SomeUnknownReturnType f() {";
        final String methodSuffix = "return noSuchReturnValue; }";

        String content = code;
        CompilationUnit parsed;
        try {
            parsed = JavaParser.parse(content);
        } catch (ParseProblemException e1) {
            // Wrap with a class and method
            try {
                content = classPrefix + methodPrefix + code + methodSuffix + classSuffix;
                parsed = JavaParser.parse(content);
            } catch (ParseProblemException e2) {
                // Wrap with a class only
                content = classPrefix + code + classSuffix;
                parsed = JavaParser.parse(content);
            }
        }

        return parsed;
    }
}
