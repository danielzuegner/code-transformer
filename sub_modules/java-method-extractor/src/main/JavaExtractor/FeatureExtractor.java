package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.MethodContent;
import JavaExtractor.FeaturesEntities.ProgramFeatures;
import JavaExtractor.Visitors.FunctionVisitor;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;

import java.util.ArrayList;

class FeatureExtractor {
    private final CommandLineValues m_CommandLineValues;

    public FeatureExtractor(CommandLineValues commandLineValues) {
        this.m_CommandLineValues = commandLineValues;
    }


    public ArrayList<ProgramFeatures> extractFeatures(String code) {
        CompilationUnit m_CompilationUnit = parseFileWithRetries(code);
        FunctionVisitor functionVisitor = new FunctionVisitor(m_CommandLineValues);

        functionVisitor.visit(m_CompilationUnit, null);

        ArrayList<MethodContent> methods = functionVisitor.getMethodContents();

        return null;
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
