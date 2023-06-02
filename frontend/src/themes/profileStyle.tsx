import {
  Avatar,
  AvatarProps,
  TableCell,
  TableCellProps,
  TextField,
  TextFieldProps
} from '@mui/material'
import { createTheme } from '@mui/material'
import { styled } from '@mui/material/styles'

export const AvatarProfile = styled(Avatar)<AvatarProps>(() => ({
  width: 110,
  height: 110,
  background: '#E7E9ED',
  border: '2px solid var(--primary-dark-color)'
}))

export const TableCellProfile = styled(TableCell)<TableCellProps>(() => ({
  padding: '9px'
}))

export const TextFieldProfile = styled(TextField)<TextFieldProps>(() => ({
  '& fieldset': {
    borderRadius: '40px'
  },
  '& .MuiInputBase-root': {
    paddingRight: 0
  },
  '& label.Mui-focused': {
    color: 'black'
  },
  '& .MuiOutlinedInput-root': {
    '&.Mui-focused fieldset': {
      borderColor: 'black'
    },
    '&.Mui-focused input': {
      borderColor: 'black'
    }
  },
  '& input': {
    padding: '3px 14px',
    borderRight: '1px solid',
    borderColor: 'rgb(192, 192, 192)'
  },
  '& .MuiInputAdornment-root': {
    cursor: 'pointer',
    margin: 0
  },
  '& .MuiButtonBase-root': {
    borderRadius: '0 30px 30px 0',
    padding: '2px 10px'
  },

  marginBottom: '10px',
  width: '230px'
}))

export const customFontIconHeader = createTheme({
  components: {
    MuiSvgIcon: {
      variants: [
        {
          props: { fontSize: 'small' },
          style: {
            fontSize: '14px'
          }
        },
        {
          props: { fontSize: 'medium' },
          style: {
            fontSize: '60px'
          }
        }
      ]
    }
  }
})
