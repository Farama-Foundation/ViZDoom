package vizdoom;
public class FileDoesNotExistException extends Exception {
    public FileDoesNotExistException(String message) {
        super(message);
    }
}
