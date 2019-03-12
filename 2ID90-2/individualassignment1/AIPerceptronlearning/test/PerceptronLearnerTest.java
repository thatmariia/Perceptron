import java.util.List;
import static org.junit.Assert.*;
import org.junit.Test;

public class PerceptronLearnerTest {
    /**
     * Example test of execute method of class PerceptronLearner.
     */
    @Test
    public void testExecute0() {
        System.out.println("execute0");
        
        //create the positive list
        List<PVector> pos = PVector.listOf(new Integer[][]{
            {2,  1}, {2,  3}, {3, -1}, {3,  2}
        });
        
        //create the negative list
        List<PVector> neg = PVector.listOf(new Integer[][]{
            {0, -2}, {1,  0}, {2, -1}, {3, -3}
        });
        
        //create the queries list
        List<PVector> q = PVector.listOf(new Integer[][]{
                {0, -2}, {1,  0}, {2, -1}, {3, -3},
                {2,  1}, {2,  3}, {3, -1}, {3,  2},
        });
        
        PerceptronLearner instance = new PerceptronLearner();
        
        //the expected result
        String expResult = "4 ----++++";
        
        //the actual result
        boolean useBias=true;
        int maxNumberOfIterations=1000;
        String result = instance.execute(pos, neg, useBias, maxNumberOfIterations, q);
        
        //test whether both are equal
        assertEquals(expResult, result);
    }
    
    @Test
    public void testExecute1() { // a template for an additional test
        System.out.println("execute1");
        
        //create the positive list
        List<PVector> pos = PVector.listOf(new Integer[][]{
            {2, 2}, {1, 4}, {4, 1}, {4, 3}, {1, 3}, {0,-123}
        });
        
        //create the negative list
        List<PVector> neg = PVector.listOf(new Integer[][]{
            {-1, -1}, {-2,  -3}, {-3, 0}, {-1, 1}, {1, -2}
        });
        
        //create the queries list
        List<PVector> q = PVector.listOf(new Integer[][]{
            {3, 1}, {1, 6}, {-3,-1}, {2, 3}, {-1, -2}, { -3, 1}, {4, 3}
        });
        
        PerceptronLearner instance = new PerceptronLearner();
        
        //the expected result
        String expResult = "3";
        
        //the actual result
        boolean useBias=false;
        int maxNumberOfIterations=3;
        String result = instance.execute(pos, neg, useBias, maxNumberOfIterations, q);
        
        //test whether both are equal
        assertEquals(expResult, result);
    }
}
