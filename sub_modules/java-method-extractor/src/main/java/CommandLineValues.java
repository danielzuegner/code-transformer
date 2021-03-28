import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class CommandLineValues {
    @Option(name = "--file", required = false)
    public java.io.File File = null;

    @Option(name = "--dir", required = false, forbids = "--file")
    public String Dir = null;

    @Option(name = "--output_dir", required = false)
    public String OutputDir = null;

    @Option(name = "--num_threads", required = false)
    public int NumThreads = 64;

    @Option(name = "--max_file_len", required = false)
    public int MaxFileLength = -1;

    @Option(name = "--min_code_len", required = false)
    public int MinCodeLength = 1;

    @Option(name = "--max_code_len", required = false)
    public int MaxCodeLength = -1;

    public CommandLineValues(String... args) throws CmdLineException {
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            throw e;
        }
    }

    public CommandLineValues() {

    }
}